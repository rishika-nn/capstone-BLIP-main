"""
Video Frame Extraction and Redundancy Filtering Module
"""

import cv2
import numpy as np
from PIL import Image
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from tqdm import tqdm
import hashlib
from collections import OrderedDict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FrameData:
    """Data structure for storing frame information"""
    frame_id: str
    timestamp: float
    frame_index: int
    image: Image.Image
    histogram: Optional[np.ndarray] = None
    phash: Optional[str] = None
    namespace: str = ""  # For object-based namespace organization
    video_date: str = ""  # Date when video was recorded (YYYY-MM-DD format)
    thumbnail_path: Optional[str] = None

class VideoFrameExtractor:
    """Extract and filter frames from video files"""
    
    def __init__(self, 
                 similarity_threshold: float = 0.85,
                 max_frames: int = 1000,
                 resize_width: Optional[int] = 640,
                 min_frames: int = 10):
        """
        Initialize the frame extractor
        
        Args:
            similarity_threshold: Threshold for frame similarity (0-1)
            max_frames: Maximum number of frames to extract
            resize_width: Width to resize frames (None for original size)
            min_frames: Minimum number of frames to extract
        """
        self.similarity_threshold = similarity_threshold
        self.max_frames = max_frames
        self.resize_width = resize_width
        self.min_frames = min_frames
        self.frames_data = []
        self.video_date = ""  # Store video date for this extraction session
        
    def extract_frames(self, video_path: str, 
                      use_similarity_filter: bool = True,
                      dedupe_method: str = 'hist',
                      interval_seconds: Optional[float] = None,
                      video_date: str = "") -> List[FrameData]:
        """
        Extract frames from video with optional redundancy filtering
        
        Args:
            video_path: Path to video file
            use_similarity_filter: Whether to use similarity-based filtering
            interval_seconds: Extract frame every N seconds (if not using similarity filter)
            video_date: Date when video was recorded (YYYY-MM-DD format)
            
        Returns:
            List of FrameData objects
        """
        # Store video date for this session
        self.video_date = video_date
        
        logger.info(f"Starting frame extraction from: {video_path}")
        if video_date:
            logger.info(f"Video date: {video_date}")
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        
        # Get video metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Video metadata - FPS: {fps:.2f}, Total frames: {total_frames}, "
                   f"Resolution: {width}x{height}, Duration: {duration:.2f}s")
        
        # Extract frames (support histogram or CLIP-based dedupe)
        if use_similarity_filter:
            if dedupe_method == 'clip':
                frames = self._extract_with_clip_dedupe(cap, fps, total_frames,
                                                       clip_threshold=self.CLIP_DEDUPE_THRESHOLD if hasattr(self, 'CLIP_DEDUPE_THRESHOLD') else 0.88)
            else:
                frames = self._extract_with_similarity_filter(cap, fps, total_frames)
        else:
            frames = self._extract_with_interval(cap, fps, total_frames, interval_seconds)
        
        cap.release()
        
        logger.info(f"Extracted {len(frames)} frames from {total_frames} total frames "
                   f"({(1 - len(frames)/total_frames)*100:.1f}% reduction)")
        
        self.frames_data = frames
        return frames
    
    def _extract_with_similarity_filter(self, cap, fps: float, 
                                       total_frames: int) -> List[FrameData]:
        """Extract frames using similarity-based redundancy filtering"""
        frames = []
        prev_histogram = None
        frame_count = 0
        
        # Create progress bar
        pbar = tqdm(total=total_frames, desc="Extracting frames")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            pbar.update(1)
            
            # Calculate timestamp
            timestamp = frame_count / fps
            
            # Resize frame if needed
            if self.resize_width and frame.shape[1] != self.resize_width:
                aspect_ratio = frame.shape[0] / frame.shape[1]
                new_height = int(self.resize_width * aspect_ratio)
                frame = cv2.resize(frame, (self.resize_width, new_height))
            
            # Calculate histogram for similarity comparison
            histogram = self._calculate_histogram(frame)
            
            # Check similarity with previous frame
            if prev_histogram is not None:
                similarity = self._compare_histograms(prev_histogram, histogram)
                if similarity > self.similarity_threshold:
                    # Skip similar frame
                    continue
            
            # Convert to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Create frame data
            frame_id = self._generate_frame_id(frame_count, timestamp)
            frame_data = FrameData(
                frame_id=frame_id,
                timestamp=timestamp,
                frame_index=frame_count,
                image=pil_image,
                histogram=histogram,
                video_date=self.video_date
            )
            
            frames.append(frame_data)
            prev_histogram = histogram
            
            # Check max frames limit
            if len(frames) >= self.max_frames:
                logger.warning(f"Reached maximum frame limit: {self.max_frames}")
                break
        
        pbar.close()
        return frames
    
    def _extract_with_interval(self, cap, fps: float, 
                              total_frames: int, 
                              interval_seconds: Optional[float]) -> List[FrameData]:
        """Extract frames at fixed time intervals"""
        frames = []
        
        # Default to 1 second if not specified
        if interval_seconds is None:
            interval_seconds = 1.0
        
        frame_interval = int(fps * interval_seconds)
        frame_count = 0
        
        # Create progress bar
        pbar = tqdm(total=total_frames, desc="Extracting frames")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            pbar.update(1)
            
            # Only keep frames at intervals
            if frame_count % frame_interval != 0:
                continue
            
            # Calculate timestamp
            timestamp = frame_count / fps
            
            # Resize frame if needed
            if self.resize_width and frame.shape[1] != self.resize_width:
                aspect_ratio = frame.shape[0] / frame.shape[1]
                new_height = int(self.resize_width * aspect_ratio)
                frame = cv2.resize(frame, (self.resize_width, new_height))
            
            # Convert to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Create frame data
            frame_id = self._generate_frame_id(frame_count, timestamp)
            frame_data = FrameData(
                frame_id=frame_id,
                timestamp=timestamp,
                frame_index=frame_count,
                image=pil_image,
                video_date=self.video_date
            )
            
            frames.append(frame_data)
            
            # Check max frames limit
            if len(frames) >= self.max_frames:
                logger.warning(f"Reached maximum frame limit: {self.max_frames}")
                break
        
        pbar.close()
        return frames
    
    def _calculate_histogram(self, frame: np.ndarray) -> np.ndarray:
        """Calculate color histogram for frame similarity comparison"""
        # Calculate histogram for each color channel
        hist_b = cv2.calcHist([frame], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([frame], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([frame], [2], None, [256], [0, 256])
        
        # Concatenate and normalize
        histogram = np.concatenate([hist_b, hist_g, hist_r]).flatten()
        histogram = histogram / histogram.sum()
        
        return histogram
    
    def _compare_histograms(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        """Compare two histograms using correlation coefficient"""
        return cv2.compareHist(hist1.astype(np.float32), 
                              hist2.astype(np.float32), 
                              cv2.HISTCMP_CORREL)
    
    def _generate_frame_id(self, frame_index: int, timestamp: float) -> str:
        """Generate unique frame ID"""
        # Create unique ID based on frame index and timestamp
        id_string = f"frame_{frame_index}_{timestamp:.3f}"
        return hashlib.md5(id_string.encode()).hexdigest()[:12]
    
    def calculate_perceptual_hash(self, image: Image.Image, hash_size: int = 8) -> str:
        """
        Calculate perceptual hash for image (alternative to histogram comparison)
        
        Args:
            image: PIL Image
            hash_size: Size of the hash
            
        Returns:
            Perceptual hash as hex string
        """
        # Resize image to hash_size x hash_size
        resized = image.resize((hash_size + 1, hash_size), Image.LANCZOS)
        
        # Convert to grayscale
        gray = resized.convert('L')
        pixels = np.array(gray)
        
        # Calculate differences between adjacent pixels
        diff = pixels[:, 1:] > pixels[:, :-1]
        
        # Convert to hash
        return hashlib.md5(diff.tobytes()).hexdigest()
    
    def filter_duplicate_captions(self, frames_with_captions: List[Tuple[FrameData, str]], 
                                 similarity_threshold: float = 0.9) -> List[Tuple[FrameData, str]]:
        """
        Filter frames with duplicate or very similar captions
        
        Args:
            frames_with_captions: List of (FrameData, caption) tuples
            similarity_threshold: Threshold for caption similarity
            
        Returns:
            Filtered list with unique captions
        """
        filtered = []
        seen_captions = set()
        
        for frame_data, caption in frames_with_captions:
            # Simple duplicate check (can be enhanced with fuzzy matching)
            caption_lower = caption.lower().strip()
            
            if caption_lower not in seen_captions:
                filtered.append((frame_data, caption))
                seen_captions.add(caption_lower)
        
        logger.info(f"Filtered {len(frames_with_captions) - len(filtered)} duplicate captions")
        return filtered
    
    def get_frame_by_timestamp(self, timestamp: float) -> Optional[FrameData]:
        """Get frame data for a specific timestamp"""
        for frame_data in self.frames_data:
            if abs(frame_data.timestamp - timestamp) < 0.1:  # Within 0.1 second
                return frame_data
        return None
    
    def get_timestamp_mapping(self) -> Dict[str, float]:
        """Get mapping of frame IDs to timestamps"""
        return {frame.frame_id: frame.timestamp for frame in self.frames_data}
    
    def save_frames_to_disk(self, output_dir: str = './extracted_frames'):
        """Save extracted frames to disk for debugging/review"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for frame_data in tqdm(self.frames_data, desc="Saving frames"):
            filename = f"{output_dir}/frame_{frame_data.frame_index:06d}_t{frame_data.timestamp:.2f}.jpg"
            frame_data.image.save(filename)
            # Also write thumbnail and attach path
            try:
                thumb_dir = os.path.join(output_dir, 'thumbnails')
                os.makedirs(thumb_dir, exist_ok=True)
                thumb_path = os.path.join(thumb_dir, f"thumb_{frame_data.frame_index:06d}.jpg")
                frame_data.image.copy().resize((256, 256)).save(thumb_path)
                frame_data.thumbnail_path = thumb_path
            except Exception:
                # Non-fatal: continue saving main images
                pass
        
        logger.info(f"Saved {len(self.frames_data)} frames to {output_dir}")

    def _extract_with_clip_dedupe(self, cap, fps: float, total_frames: int, clip_threshold: float = 0.88) -> List[FrameData]:
        """Extract frames using CLIP semantic deduplication"""
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            logger.error("sentence-transformers is required for CLIP dedupe: %s", e)
            # Fall back to histogram method
            return self._extract_with_similarity_filter(cap, fps, total_frames)

        # Lazy-load CLIP model (text+image support)
        # Try loading a CLIP model with common ids and fall back gracefully
        clip_model = None
        tried = []
        for candidate in [getattr(self, 'CLIP_MODEL_NAME', None), 'clip-ViT-B-32', 'openai/clip-vit-base-patch32']:
            if not candidate:
                continue
            try:
                clip_model = SentenceTransformer(candidate)
                logger.info(f"Loaded CLIP model for dedupe: {candidate}")
                break
            except Exception as e:
                tried.append((candidate, str(e)))
                logger.warning(f"Failed to load CLIP model '{candidate}': {e}")

        if clip_model is None:
            logger.error(f"Could not load any CLIP model for dedupe. Tried: {[c[0] for c in tried]}")
            # Fall back to histogram dedupe
            return self._extract_with_similarity_filter(cap, fps, total_frames)

        kept_frames: List[FrameData] = []
        kept_embeddings = []  # normalized

        frame_count = 0
        pbar = tqdm(total=total_frames, desc="Extracting frames (CLIP dedupe)")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            pbar.update(1)

            # Resize for efficiency
            if self.resize_width and frame.shape[1] != self.resize_width:
                aspect_ratio = frame.shape[0] / frame.shape[1]
                new_height = int(self.resize_width * aspect_ratio)
                frame = cv2.resize(frame, (self.resize_width, new_height))

            # Convert to PIL image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # Create frame data
            timestamp = frame_count / fps
            frame_id = self._generate_frame_id(frame_count, timestamp)
            frame_data = FrameData(
                frame_id=frame_id,
                timestamp=timestamp,
                frame_index=frame_count,
                image=pil_image,
                video_date=self.video_date
            )

            # Compute embedding for this frame
            try:
                emb = clip_model.encode(pil_image, convert_to_numpy=True, normalize_embeddings=True)
            except Exception:
                # If image encoding fails, fallback to histogram method decision
                histogram = self._calculate_histogram(frame)
                if not kept_frames:
                    frame_data.histogram = histogram
                    kept_frames.append(frame_data)
                else:
                    sim = self._compare_histograms(kept_frames[-1].histogram, histogram) if kept_frames[-1].histogram is not None else 0
                    if sim < self.similarity_threshold:
                        frame_data.histogram = histogram
                        kept_frames.append(frame_data)
                if len(kept_frames) >= self.max_frames:
                    break
                else:
                    continue

            # Compare with kept embeddings
            if not kept_embeddings:
                kept_embeddings.append(emb)
                kept_frames.append(frame_data)
            else:
                kept_matrix = np.stack(kept_embeddings, axis=0)
                # cosine since embeddings normalized -> dot product
                sim_scores = np.dot(kept_matrix, emb)
                max_sim = float(np.max(sim_scores))
                if max_sim < clip_threshold:
                    kept_embeddings.append(emb)
                    kept_frames.append(frame_data)

            if len(kept_frames) >= self.max_frames:
                logger.warning(f"Reached maximum frame limit: {self.max_frames}")
                break

        pbar.close()
        return kept_frames


# Example usage and testing
if __name__ == "__main__":
    # Test the frame extractor
    extractor = VideoFrameExtractor(
        similarity_threshold=0.85,
        max_frames=100,
        resize_width=640
    )
    
    # Example: Extract frames from a video
    # frames = extractor.extract_frames("sample_video.mp4")
    # print(f"Extracted {len(frames)} frames")
    # 
    # # Get timestamp mapping
    # timestamp_map = extractor.get_timestamp_mapping()
    # print(f"First 5 timestamps: {list(timestamp_map.values())[:5]}")