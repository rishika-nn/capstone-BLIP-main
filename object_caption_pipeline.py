"""
Object-Focused Captioning Pipeline
Combines Grounding DINO object detection with BLIP attribute-based captioning
Designed for campus surveillance: focuses on objects and their attributes, not actions
"""

import torch
from PIL import Image
import logging
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
from difflib import SequenceMatcher

from object_detector import GroundingDINODetector, DetectedObject
from caption_generator import BlipCaptionGenerator
from frame_extractor import FrameData

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ObjectCaption:
    """Data structure for object-focused captions"""
    frame_data: FrameData
    object_label: str  # From Grounding DINO
    object_bbox: Tuple[int, int, int, int]
    attribute_caption: str  # From BLIP (focused on attributes)
    confidence: float
    is_object_focused: bool = True  # Flag to distinguish from scene captions
    namespace: str = ""  # Pinecone namespace for this object category

class ObjectCaptionPipeline:
    """
    Pipeline that detects objects using Grounding DINO and generates 
    attribute-focused captions using BLIP
    """
    
    def __init__(self,
                 object_detector: Optional[GroundingDINODetector] = None,
                 caption_generator: Optional[BlipCaptionGenerator] = None,
                 use_gpu: bool = True,
                 min_object_size: int = 20,
                 max_objects_per_frame: int = 10,
                 include_scene_caption: bool = False,
                 caption_similarity_threshold: float = 0.85):
        """
        Initialize the object-focused captioning pipeline
        
        Args:
            object_detector: Grounding DINO detector instance (creates new if None)
            caption_generator: BLIP caption generator (creates new if None)
            use_gpu: Whether to use GPU
            min_object_size: Minimum object size (pixels) to caption
            max_objects_per_frame: Maximum objects to caption per frame
            include_scene_caption: Whether to also generate full scene caption
            caption_similarity_threshold: Threshold for filtering similar captions (0-1)
        """
        self.use_gpu = use_gpu
        self.min_object_size = min_object_size
        self.max_objects_per_frame = max_objects_per_frame
        self.include_scene_caption = include_scene_caption
        self.caption_similarity_threshold = caption_similarity_threshold
        
        # Track unique captions to prevent duplicates
        self._seen_captions: Set[str] = set()
        self._caption_history: List[str] = []
        
        # Initialize or use provided components
        if object_detector is None:
            logger.info("Initializing Grounding DINO detector...")
            # Slightly lower threshold to catch smaller/farther objects typical in surveillance
            self.object_detector = GroundingDINODetector(
                confidence_threshold=0.22,
                use_gpu=use_gpu
            )
        else:
            self.object_detector = object_detector
        
        if caption_generator is None:
            logger.info("Initializing BLIP caption generator...")
            self.caption_generator = BlipCaptionGenerator(
                batch_size=4,  # Smaller batch for cropped objects
                use_gpu=use_gpu,
                max_length=30,  # Shorter captions for objects
                num_beams=3
            )
        else:
            self.caption_generator = caption_generator
        
        # Attribute-focused prompts for BLIP
        # Tuned to elicit attributes (color, material, style) instead of actions
        self.attribute_prompts = [
            "a close-up photo of the {label}. Describe color, material, and notable attributes",
            "describe the {label} focusing on color, style, and accessories",
            "{label}. Provide a short attribute-focused description (color, size, pattern)"
        ]

        # Default object prompts for classroom/surveillance contexts
        self.default_object_prompts = [
            # People
            "person", "man", "woman",
            # Bags and items often left on ground
            "bag", "backpack", "duffel bag", "suitcase", "luggage", "bag on ground", "backpack on ground",
            # Clothing/outerwear
            "jacket", "coat",
            # Personal electronics
            "laptop", "tablet", "phone",
            # Bottles/umbrellas
            "bottle", "umbrella",
            # Classroom fixtures
            "desk", "chair", "bench", "whiteboard", "projector", "screen"
        ]
        
        # Common color words to help score candidate captions
        self._color_words = set([
            'black','white','gray','grey','red','orange','yellow','green','blue','purple','pink',
            'brown','beige','tan','gold','silver','navy','maroon','teal'
        ])
        
        # Words that indicate person-focused descriptions (to filter out)
        self._person_indicators = set([
            'person', 'people', 'man', 'woman', 'student', 'wearing', 'carrying',
            'holding', 'walking', 'standing', 'sitting', 'running', 'talking'
        ])
        
        logger.info("Object-focused captioning pipeline initialized")
    
    def process_frame(self,
                     frame_data: FrameData,
                     object_prompts: Optional[List[str]] = None,
                     verbose: bool = True) -> List[ObjectCaption]:
        """
        Process a single frame: detect objects and caption them
        
        Args:
            frame_data: Frame to process
            object_prompts: Specific objects to detect (uses defaults if None)
            verbose: Whether to print detailed logging for each caption
            
        Returns:
            List of ObjectCaption instances
        """
        object_captions = []
        
        try:
            # Step 1: Detect objects using Grounding DINO
            # Use default prompts when none provided (classroom/surveillance tuned)
            prompts = object_prompts or self.default_object_prompts
            detections = self.object_detector.detect_objects(
                image=frame_data.image,
                text_prompts=prompts,
                return_crops=True
            )
            
            # Filter small objects and limit to top detections
            detections = self._filter_detections(detections)
            
            if not detections:
                if verbose:
                    print(f"\n📸 Processing Frame: {frame_data.frame_id} (t={frame_data.timestamp:.2f}s)")
                    print(f"   No objects detected (after filtering)")
                logger.debug(f"No objects detected in frame {frame_data.frame_id}")
                
                # Optionally generate scene caption as fallback
                if self.include_scene_caption:
                    scene_caption = self._generate_scene_caption(frame_data)
                    if scene_caption:
                        object_captions.append(scene_caption)
                
                return object_captions
            
            # Step 2: Generate attribute-focused captions for each detected object
            if verbose:
                print(f"\n📸 Processing Frame: {frame_data.frame_id} (t={frame_data.timestamp:.2f}s)")
                print(f"   Found {len(detections)} objects")
            
            for idx, detection in enumerate(detections, 1):
                if detection.cropped_image is None:
                    continue
                # Expand crop a bit to include context (straps, edges)
                expanded_crop = self._expanded_crop(frame_data.image, detection.bbox, expand_ratio=0.15)

                # Generate multiple candidate captions and pick best
                candidates = []
                cap1 = self._caption_object(detection.cropped_image, detection.label)
                if cap1:
                    candidates.append(cap1)
                if expanded_crop is not None:
                    cap2 = self._caption_object(expanded_crop, detection.label)
                    if cap2:
                        candidates.append(cap2)

                caption_text = None
                if candidates:
                    # Score and choose best candidate
                    scored = [(self._score_caption(c, detection.label), c) for c in candidates]
                    scored.sort(key=lambda x: x[0], reverse=True)
                    caption_text = scored[0][1]
                
                # Validate caption quality and uniqueness
                if caption_text and len(caption_text.split()) >= 4:  # Increased minimum
                    # Check if caption is truly unique
                    if self._is_unique_caption(caption_text):
                        # Determine namespace for this object category
                        namespace = self._get_namespace_for_object(detection.label)
                        
                        obj_caption = ObjectCaption(
                            frame_data=frame_data,
                            object_label=detection.label,
                            object_bbox=detection.bbox,
                            attribute_caption=caption_text,
                            confidence=detection.confidence,
                            is_object_focused=True,
                            namespace=namespace
                        )
                        object_captions.append(obj_caption)
                        
                        # Track this caption to prevent future duplicates
                        self._add_to_caption_history(caption_text)
                        
                        # Real-time logging
                        if verbose:
                            print(f"   ├─ Object {idx}: {detection.label.title()}")
                            print(f"   │  Caption: {caption_text}")
                            print(f"   │  Namespace: {namespace}")
                            print(f"   │  Confidence: {detection.confidence:.2%}")
                    else:
                        if verbose:
                            print(f"   ├─ Object {idx}: {detection.label.title()} (skipped - duplicate)")
                        logger.debug(f"Skipping duplicate caption: '{caption_text}'")
                else:
                    if verbose:
                        reason = "failed validation" if caption_text else "no caption generated"
                        print(f"   ├─ Object {idx}: {detection.label.title()} (skipped - {reason})")
                    logger.debug(f"Skipping invalid caption: '{caption_text}'")
            
            if verbose and object_captions:
                print(f"   └─ ✓ Generated {len(object_captions)} valid caption(s)")
            elif verbose and not object_captions and detections:
                print(f"   └─ ⚠️  No valid captions generated (all failed validation)")
            
            logger.debug(f"Generated {len(object_captions)} object captions for frame {frame_data.frame_id}")
            
        except Exception as e:
            logger.error(f"Error processing frame {frame_data.frame_id}: {e}")
        
        return object_captions
    
    def process_frames(self,
                      frames: List[FrameData],
                      object_prompts: Optional[List[str]] = None,
                      show_progress: bool = True) -> List[ObjectCaption]:
        """
        Process multiple frames
        
        Args:
            frames: List of frames to process
            object_prompts: Specific objects to detect
            show_progress: Whether to show progress bar
            
        Returns:
            List of all ObjectCaption instances from all frames
        """
        all_captions = []
        
        iterator = tqdm(frames, desc="Processing frames") if show_progress else frames
        
        for frame_data in iterator:
            frame_captions = self.process_frame(frame_data, object_prompts)
            all_captions.extend(frame_captions)
        
        logger.info(f"Generated {len(all_captions)} total object captions from {len(frames)} frames")
        return all_captions
    
    def _filter_detections(self, detections: List[DetectedObject]) -> List[DetectedObject]:
        """Filter detections by size and limit to top K"""
        # Filter by minimum size
        filtered = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            width = x2 - x1
            height = y2 - y1
            
            if width >= self.min_object_size and height >= self.min_object_size:
                filtered.append(det)
        
        # Get top K by confidence
        filtered = self.object_detector.get_top_detections(
            filtered,
            top_k=self.max_objects_per_frame
        )
        
        return filtered
    
    def _caption_object(self, cropped_image: Image.Image, object_label: str) -> str:
        """
        Generate attribute-focused caption for a detected object
        
        Args:
            cropped_image: Cropped object image
            object_label: Object label from Grounding DINO
            
        Returns:
            Attribute-focused caption string
        """
        try:
            # Generate caption for the cropped object (unconditional generation)
            # BLIP works better without text prompts for image captioning
            with torch.no_grad():
                inputs = self.caption_generator.processor(
                    images=cropped_image,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.caption_generator.device) for k, v in inputs.items()}
                outputs = self.caption_generator.model.generate(
                    **inputs,
                    max_length=40,  # Increased for complete sentences
                    num_beams=5,    # More beams for better quality
                    do_sample=False,
                    early_stopping=True,
                    repetition_penalty=1.2,  # Reduce repetitive words
                    length_penalty=1.0
                )
                raw_caption = self.caption_generator.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Clean the raw caption to remove special characters
            raw_caption = self._clean_raw_caption(raw_caption)
            
            # Verify object is actually described in caption
            if not self._verify_object_in_caption(raw_caption, object_label):
                logger.debug(f"Object '{object_label}' not found in caption: '{raw_caption}'")
                return None
            
            # Format with object label into grammatically correct sentence
            formatted_caption = self._format_grammatical_caption(raw_caption, object_label)
            
            # Validate grammatical correctness
            if not self._is_grammatically_valid(formatted_caption):
                logger.debug(f"Caption failed grammar check: '{formatted_caption}'")
                return None
            
            return formatted_caption
            
        except Exception as e:
            logger.error(f"Error captioning object {object_label}: {e}")
            return None

    def _expanded_crop(self, image: Image.Image, bbox: Tuple[int, int, int, int], expand_ratio: float = 0.15) -> Optional[Image.Image]:
        """Return an expanded crop around bbox to capture more context."""
        try:
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            dw, dh = int(w * expand_ratio), int(h * expand_ratio)
            nx1 = max(0, x1 - dw)
            ny1 = max(0, y1 - dh)
            nx2 = min(image.width, x2 + dw)
            ny2 = min(image.height, y2 + dh)
            if nx2 - nx1 <= 1 or ny2 - ny1 <= 1:
                return None
            return image.crop((nx1, ny1, nx2, ny2))
        except Exception:
            return None

    def _score_caption(self, caption: str, object_label: str) -> float:
        """Heuristic score combining color words, label presence and length."""
        if not caption:
            return 0.0
        c = caption.lower()
        score = 0.0
        # Reward color/material descriptors
        for w in [
            'black','white','grey','gray','red','blue','green','brown','beige','tan','leather','fabric','nylon'
        ]:
            if w in c:
                score += 0.5
        # Reward label/synonyms presence
        syn = {
            'backpack': ['backpack','bag','pack'],
            'bag': ['bag','backpack','satchel','purse'],
            'suitcase': ['suitcase','luggage'],
            'jacket': ['jacket','coat'],
            'coat': ['coat','jacket'],
            'person': ['person','man','woman']
        }
        label_terms = syn.get(object_label.lower(), [object_label.lower()])
        if any(t in c for t in label_terms):
            score += 1.0
        # Length preference
        n = len(caption.split())
        if 6 <= n <= 20:
            score += 0.5
        # Grammar check reuse
        if self._is_grammatically_valid(caption):
            score += 0.5
        return score
    
    def _format_attribute_caption(self, blip_caption: str, object_label: str) -> str:
        """
        DEPRECATED: Use _format_grammatical_caption instead
        Kept for backward compatibility
        """
        return self._format_grammatical_caption(blip_caption, object_label)
    
    def _format_grammatical_caption(self, blip_caption: str, object_label: str) -> str:
        """
        Format BLIP caption into a grammatically correct, complete sentence
        
        Args:
            blip_caption: Raw caption from BLIP
            object_label: Object label from detector
            
        Returns:
            Grammatically correct caption
        """
        # Clean and validate object label
        object_label = self._clean_object_label(object_label)
        
        # Clean caption
        caption = (blip_caption or "").strip()
        
        if not caption:
            return None
        
        # Extract object type (normalize label)
        obj_type = object_label.replace('_', ' ').title()
        
        # Check if caption already starts with a determiner
        caption_lower = caption.lower()
        
        # Build grammatically correct sentence
        # Pattern: "{Object}: A/An {color} {object} {description}."
        
        # Determine article (a vs an)
        first_word = caption.split()[0].lower() if caption.split() else ""
        article = "An" if first_word and first_word[0] in 'aeiou' else "A"
        
        # Check if caption already contains the object type
        obj_in_caption = any(word in caption_lower for word in object_label.lower().split())
        
        if obj_in_caption:
            # Caption already mentions the object
            # Example: "red backpack on a person" -> "Backpack: A red backpack on a person."
            if caption[0].islower():
                caption = caption[0].upper() + caption[1:]
            formatted = f"{obj_type}: {article} {caption}"
        else:
            # Caption doesn't mention object, add it
            # Example: "red with straps" -> "Backpack: A red backpack with straps."
            formatted = f"{obj_type}: {article} {caption.lower()}"
        
        # Ensure proper ending punctuation
        if formatted and formatted[-1] not in '.!?':
            formatted += '.'
        
        # Clean up common grammar issues
        formatted = formatted.replace('  ', ' ')  # Remove double spaces
        formatted = formatted.replace(' .', '.')  # Fix space before period
        formatted = formatted.replace('a a ', 'a ')  # Remove duplicate articles
        formatted = formatted.replace('A a ', 'A ')
        formatted = formatted.replace('the the ', 'the ')  # Remove duplicate determiners
        
        return formatted
    
    def _generate_scene_caption(self, frame_data: FrameData) -> Optional[ObjectCaption]:
        """
        Generate a full scene caption as fallback
        
        Args:
            frame_data: Frame to caption
            
        Returns:
            ObjectCaption with scene description
        """
        try:
            # Generate standard BLIP caption for full frame
            inputs = self.caption_generator.processor(
                images=frame_data.image,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.caption_generator.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.caption_generator.model.generate(
                    **inputs,
                    max_length=50,
                    num_beams=4,
                    do_sample=False,
                    early_stopping=True
                )
            
            caption = self.caption_generator.processor.decode(outputs[0], skip_special_tokens=True)
            caption = self.caption_generator._clean_caption(caption)
            
            # Create scene caption object
            scene_caption = ObjectCaption(
                frame_data=frame_data,
                object_label="scene",
                object_bbox=(0, 0, frame_data.image.width, frame_data.image.height),
                attribute_caption=caption,
                confidence=1.0,
                is_object_focused=False
            )
            
            return scene_caption
            
        except Exception as e:
            logger.error(f"Error generating scene caption: {e}")
            return None
    
    def get_statistics(self, object_captions: List[ObjectCaption]) -> Dict:
        """Get statistics about generated object captions"""
        if not object_captions:
            return {"total": 0}
        
        # Count by object type
        object_counts = {}
        for oc in object_captions:
            label = oc.object_label
            object_counts[label] = object_counts.get(label, 0) + 1
        
        # Calculate average confidence
        avg_confidence = sum(oc.confidence for oc in object_captions) / len(object_captions)
        
        stats = {
            "total_captions": len(object_captions),
            "unique_frames": len(set(oc.frame_data.frame_id for oc in object_captions)),
            "object_focused": sum(1 for oc in object_captions if oc.is_object_focused),
            "scene_captions": sum(1 for oc in object_captions if not oc.is_object_focused),
            "avg_confidence": avg_confidence,
            "objects_detected": object_counts,
            "most_common_object": max(object_counts, key=object_counts.get) if object_counts else None
        }
        
        return stats
    
    def _clean_raw_caption(self, caption: str) -> str:
        """
        Clean raw caption from BLIP to remove special characters and noise
        
        Args:
            caption: Raw caption from BLIP model
            
        Returns:
            Cleaned caption text
        """
        if not caption:
            return ""
        
        import re
        
        # Remove special character prefixes (e.g., "##El:", "@@tag:", etc.)
        caption = re.sub(r'^[#@!$%^&*]+[a-zA-Z]*\s*:\s*', '', caption)
        
        # Remove standalone special characters
        caption = re.sub(r'\s+[#@!$%^&*]+\s+', ' ', caption)
        
        # Remove HTML/XML-like tags
        caption = re.sub(r'<[^>]+>', '', caption)
        
        # Keep only alphanumeric, spaces, and basic punctuation
        caption = re.sub(r'[^a-zA-Z0-9\s\.\,\-\'\"]+', ' ', caption)
        
        # Clean up multiple spaces
        caption = ' '.join(caption.split())
        
        # Remove very short tokens (likely noise)
        tokens = caption.split()
        cleaned_tokens = [t for t in tokens if len(t) > 1 or t.lower() in ['a', 'i']]
        caption = ' '.join(cleaned_tokens)
        
        return caption.strip()
    
    def _clean_object_label(self, label: str) -> str:
        """
        Clean and validate object label to remove invalid characters
        
        Args:
            label: Raw object label from detector
            
        Returns:
            Cleaned, valid object label
        """
        if not label:
            return "object"
        
        # Remove common noise patterns
        import re
        
        # Remove special characters at the start (like ##, @@, etc.)
        label = re.sub(r'^[#@!$%^&*]+', '', label)
        
        # Remove any non-alphanumeric characters except spaces and hyphens
        label = re.sub(r'[^a-zA-Z0-9\s\-]', '', label)
        
        # Clean up whitespace
        label = ' '.join(label.split())
        
        # If label is too short or empty after cleaning, use default
        if len(label) < 2:
            return "object"
        
        # Convert to lowercase for consistency
        label = label.lower().strip()
        
        # Validate against known object types
        valid_prefixes = ['bag', 'backpack', 'duffel', 'laptop', 'computer', 'tablet',
                         'helmet', 'bottle', 'water', 'folder', 'file', 'umbrella',
                         'coat', 'jacket', 'suitcase', 'luggage', 'person', 'object']
        
        # If label doesn't start with any valid prefix, use generic "object"
        if not any(label.startswith(prefix) for prefix in valid_prefixes):
            logger.debug(f"Invalid label detected: '{label}', using 'object'")
            return "object"
        
        return label
    
    def _is_unique_caption(self, caption: str) -> bool:
        """
        Check if caption is unique compared to previously seen captions
        
        Args:
            caption: Caption to check
            
        Returns:
            True if caption is sufficiently unique, False otherwise
        """
        caption_normalized = caption.lower().strip()
        
        # Check exact match first
        if caption_normalized in self._seen_captions:
            return False
        
        # Check similarity against recent captions
        for prev_caption in self._caption_history[-20:]:  # Check last 20 captions
            similarity = self._caption_similarity(caption_normalized, prev_caption.lower())
            if similarity >= self.caption_similarity_threshold:
                logger.debug(f"Caption too similar ({similarity:.2f}): '{caption}' vs '{prev_caption}'")
                return False
        
        return True
    
    def _add_to_caption_history(self, caption: str):
        """
        Add caption to history for duplicate tracking
        
        Args:
            caption: Caption to add
        """
        caption_normalized = caption.lower().strip()
        self._seen_captions.add(caption_normalized)
        self._caption_history.append(caption)
        
        # Keep history size manageable (last 50 captions)
        if len(self._caption_history) > 50:
            # Remove oldest caption from history
            oldest = self._caption_history.pop(0)
            # Keep in seen_captions set for exact match checking
    
    def _caption_similarity(self, caption1: str, caption2: str) -> float:
        """
        Calculate similarity between two captions using sequence matching
        
        Args:
            caption1: First caption
            caption2: Second caption
            
        Returns:
            Similarity score between 0 and 1
        """
        return SequenceMatcher(None, caption1, caption2).ratio()
    
    def _verify_object_in_caption(self, caption: str, object_label: str) -> bool:
        """
        Verify that the detected object is actually described in the caption
        
        Args:
            caption: Generated caption text
            object_label: Expected object label
            
        Returns:
            True if object appears to be described, False otherwise
        """
        if not caption or not object_label:
            return False
        
        caption_lower = caption.lower()
        
        # Extract key terms from object label
        label_terms = object_label.lower().replace('_', ' ').split()
        
        # Object-related keywords that should appear
        object_keywords = set(label_terms)
        
        # Also accept broader categories
        category_map = {
            'backpack': ['backpack', 'bag', 'pack'],
            'duffel': ['duffel', 'bag', 'duffle'],
            'bag': ['bag', 'backpack', 'satchel', 'purse'],
            'laptop': ['laptop', 'computer', 'notebook'],
            'tablet': ['tablet', 'ipad', 'device'],
            'helmet': ['helmet', 'hat', 'headgear'],
            'bottle': ['bottle', 'container', 'flask'],
            'folder': ['folder', 'file', 'document'],
            'umbrella': ['umbrella', 'parasol'],
            'coat': ['coat', 'jacket', 'outerwear'],
            'jacket': ['jacket', 'coat', 'outerwear'],
            'suitcase': ['suitcase', 'luggage', 'bag'],
            'luggage': ['luggage', 'suitcase', 'bag']
        }
        
        # Get acceptable terms for this object
        acceptable_terms = object_keywords.copy()
        for term in label_terms:
            if term in category_map:
                acceptable_terms.update(category_map[term])
        
        # Check if any acceptable term appears in caption
        for term in acceptable_terms:
            if term in caption_lower:
                return True
        
        # Additional check: look for visual descriptors that make sense
        # If caption has reasonable descriptive words, it might still be valid
        descriptive_words = ['red', 'blue', 'black', 'white', 'green', 'yellow',
                           'large', 'small', 'leather', 'fabric', 'metal', 'plastic']
        
        has_descriptors = any(word in caption_lower for word in descriptive_words)
        
        # If no object terms found but has descriptors, likely misidentified
        if not has_descriptors:
            logger.debug(f"Caption lacks object reference or descriptors: '{caption}' for '{object_label}'")
            return False
        
        # Last resort: check caption is not about completely different things
        irrelevant_keywords = ['car', 'vehicle', 'building', 'tree', 'sky', 'rabbit', 
                              'cat', 'dog', 'bird', 'animal']
        for keyword in irrelevant_keywords:
            if keyword in caption_lower and keyword not in object_label.lower():
                logger.debug(f"Caption describes irrelevant object '{keyword}': '{caption}'")
                return False
        
        return True
    
    def _is_grammatically_valid(self, caption: str) -> bool:
        """
        Check if caption is grammatically valid and well-formed
        
        Args:
            caption: Caption to validate
            
        Returns:
            True if caption passes basic grammar checks
        """
        if not caption:
            return False
        
        # Check minimum length (at least 4 words for a complete sentence)
        words = caption.split()
        if len(words) < 4:
            logger.debug(f"Caption too short: '{caption}'")
            return False
        
        # Check for repetitive articles or words
        # Example: "a a backpack" or "the the bag"
        for i in range(len(words) - 1):
            if words[i].lower() == words[i+1].lower() and words[i].lower() in ['a', 'an', 'the']:
                logger.debug(f"Repetitive article detected: '{caption}'")
                return False
        
        # Check for incomplete fragments (multiple consecutive single letters)
        single_letters = [w for w in words if len(w) == 1 and w.lower() not in ['a', 'i']]
        if len(single_letters) > 2:
            logger.debug(f"Too many single letter fragments: '{caption}'")
            return False
        
        # Check for proper sentence structure (should have object label prefix)
        if ':' not in caption:
            logger.debug(f"Missing object label prefix: '{caption}'")
            return False
        
        # Split by colon and check description part
        parts = caption.split(':', 1)
        if len(parts) != 2:
            return False
        
        description = parts[1].strip()
        desc_words = description.split()
        
        # Description should be reasonable length
        if len(desc_words) < 3:
            logger.debug(f"Description too short: '{description}'")
            return False
        
        # Check for proper article at start of description (relaxed - just a warning)
        first_word = desc_words[0].lower() if desc_words else ""
        if first_word not in ['a', 'an', 'the']:
            # Relaxed: Allow captions without articles if they have good content
            # Check if it has meaningful content words
            content_words = [w for w in desc_words if len(w) > 3]
            if len(content_words) < 2:
                logger.debug(f"Description missing article AND lacks content: '{description}'")
                return False
            # Has content, so we'll allow it despite missing article
            logger.debug(f"Description missing article but has content: '{description}'")
        
        # Check for reasonable word diversity (not all same word)
        unique_words = set(w.lower() for w in desc_words)
        if len(unique_words) < len(desc_words) * 0.5:  # At least 50% unique
            logger.debug(f"Low word diversity: '{description}'")
            return False
        
        return True
    
    def reset_caption_history(self):
        """Reset the caption history and seen captions (useful for processing new videos)"""
        self._seen_captions.clear()
        self._caption_history.clear()
        logger.info("Caption history reset")
    
    def _get_namespace_for_object(self, object_label: str) -> str:
        """
        Map object label to its Pinecone namespace
        
        Args:
            object_label: Object label from detector
            
        Returns:
            Namespace name (lowercase, underscore-separated)
        """
        # Normalize label
        label_lower = object_label.lower().strip()
        
        # Map similar objects to same namespace
        namespace_mapping = {
            # Bags
            'backpack': 'backpack',
            'bag': 'bag',
            'duffel bag': 'duffel_bag',
            'duffel': 'duffel_bag',
            
            # Electronics
            'laptop': 'laptop',
            'computer': 'laptop',
            'tablet': 'tablet',
            
            # Safety
            'helmet': 'helmet',
            
            # Containers
            'bottle': 'bottle',
            'water bottle': 'bottle',
            
            # Personal items
            'folder': 'folder',
            'file folder': 'folder',
            'document folder': 'folder',
            'umbrella': 'umbrella',
            
            # Clothing
            'coat': 'coat_jacket',
            'jacket': 'coat_jacket',
            'coat jacket': 'coat_jacket',
            
            # Travel
            'suitcase': 'suitcase_luggage',
            'luggage': 'suitcase_luggage',
            'suitcase luggage': 'suitcase_luggage'
        }
        
        # Try exact match first
        if label_lower in namespace_mapping:
            return namespace_mapping[label_lower]
        
        # Try partial match
        for key, namespace in namespace_mapping.items():
            if key in label_lower or label_lower in key:
                return namespace
        
        # Default: use cleaned label as namespace
        namespace = label_lower.replace(' ', '_').replace('-', '_')
        return namespace
    
    def clear_cache(self):
        """Clear GPU cache from both models"""
        self.object_detector.clear_gpu_cache()
        self.caption_generator.clear_gpu_cache()
        logger.info("Pipeline cache cleared")
    
    def unload_models(self):
        """Unload both models from memory"""
        self.object_detector.unload_model()
        self.caption_generator.unload_model()
        logger.info("Pipeline models unloaded")


# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = ObjectCaptionPipeline(use_gpu=True)
    
    # Example: Process frames
    # from frame_extractor import VideoFrameExtractor
    # 
    # extractor = VideoFrameExtractor()
    # frames = extractor.extract_frames("sample_video.mp4")
    # 
    # # Process with object-focused captioning
    # object_captions = pipeline.process_frames(frames)
    # 
    # # Print results
    # for oc in object_captions[:10]:
    #     print(f"Time: {oc.frame_data.timestamp:.2f}s")
    #     print(f"  Object: {oc.object_label} (conf: {oc.confidence:.2f})")
    #     print(f"  Caption: {oc.attribute_caption}")
    #     print()
    # 
    # # Get statistics
    # stats = pipeline.get_statistics(object_captions)
    # print(f"Statistics: {stats}")
