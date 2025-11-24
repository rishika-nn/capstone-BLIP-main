"""
Video Frame Search Engine - Main Application
Production-ready video semantic search system with BLIP and Pinecone
"""

import os
import logging
from typing import List, Optional, Dict, Tuple, Any
from pathlib import Path
import json
import time
from datetime import datetime
import pandas as pd
import numpy as np

# Import all modules
from video_search_config import Config
from frame_extractor import VideoFrameExtractor, FrameData
from caption_generator import BlipCaptionGenerator, CaptionedFrame
from embedding_generator import TextEmbeddingGenerator, EmbeddedFrame, MultimodalEmbeddingGenerator
from pinecone_manager import PineconeManager, SearchResult
from object_caption_pipeline import ObjectCaptionPipeline, ObjectCaption
from temporal_bootstrapping import TemporalBootstrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video_search_engine.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VideoSearchEngine:
    """
    Complete video search engine integrating all components
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the video search engine
        
        Args:
            config: Configuration object (uses default if None)
        """
        # Use provided config or default
        self.config = config or Config()
        
        # Validate configuration
        try:
            self.config.validate()
        except ValueError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
        
        # Initialize components (lazy loading)
        self.frame_extractor = None
        self.caption_generator = None
        self.embedding_generator = None
        self.pinecone_manager = None
        self.object_pipeline = None  # Object-focused captioning pipeline
        
        # Processing state
        self.current_video = None
        self.processed_frames = []
        self.processing_stats = {}
        
        logger.info("Video Search Engine initialized")
    
    def _initialize_components(self):
        """Initialize all components if not already initialized"""
        if not self.frame_extractor:
            self.frame_extractor = VideoFrameExtractor(
                similarity_threshold=self.config.FRAME_SIMILARITY_THRESHOLD,
                max_frames=self.config.MAX_FRAMES_PER_VIDEO,
                resize_width=self.config.FRAME_RESIZE_WIDTH
            )
            logger.info("Frame extractor initialized")
        
        if not self.caption_generator:
            self.caption_generator = BlipCaptionGenerator(
                model_name=self.config.BLIP_MODEL,
                batch_size=self.config.BLIP_BATCH_SIZE,
                use_gpu=self.config.USE_GPU,
                generate_multiple_captions=getattr(self.config, 'GENERATE_MULTIPLE_CAPTIONS', False),
                captions_per_frame=getattr(self.config, 'CAPTIONS_PER_FRAME', 3),
                enable_clip_rerank=getattr(self.config, 'ENABLE_CLIP_RERANK', False),
                clip_rerank_model=getattr(self.config, 'CLIP_RERANK_MODEL', None)
            )
            logger.info("Caption generator initialized with multi-caption support")
        
        if not self.embedding_generator:
            # Prefer multimodal generator when dual embeddings are enabled
            if getattr(self.config, 'ENABLE_DUAL_EMBEDDINGS', False):
                self.embedding_generator = MultimodalEmbeddingGenerator(
                    caption_model=self.config.EMBEDDING_MODEL,
                    image_model=getattr(self.config, 'CLIP_MODEL_NAME', 'clip-ViT-B-32'),
                    batch_size=self.config.EMBEDDING_BATCH_SIZE,
                    use_gpu=self.config.USE_GPU,
                    normalize=True
                )
                logger.info("Multimodal embedding generator initialized (caption + image)")
            else:
                self.embedding_generator = TextEmbeddingGenerator(
                    model_name=self.config.EMBEDDING_MODEL,
                    batch_size=self.config.EMBEDDING_BATCH_SIZE,
                    use_gpu=self.config.USE_GPU,
                    normalize=True
                )
                logger.info("Text embedding generator initialized")
        
        if not self.pinecone_manager:
            self.pinecone_manager = PineconeManager(
                api_key=self.config.PINECONE_API_KEY,
                environment=self.config.PINECONE_ENVIRONMENT,
                index_name=self.config.PINECONE_INDEX_NAME,
                dimension=self.config.PINECONE_DIMENSION,
                metric=self.config.PINECONE_METRIC,
                host=getattr(self.config, 'PINECONE_HOST', None)
            )
            logger.info("Pinecone manager initialized")
    
    def process_video(self, 
                     video_path: str,
                     video_name: Optional[str] = None,
                     video_date: Optional[str] = None,
                     save_frames: bool = False,
                     upload_to_pinecone: bool = True,
                     use_object_detection: bool = True) -> Dict[str, Any]:
        """
        Process a video file end-to-end
        
        Args:
            video_path: Path to video file
            video_name: Name for the video (uses filename if None)
            video_date: Date when video was recorded (YYYY-MM-DD format, uses today if None)
            save_frames: Whether to save extracted frames to disk
            upload_to_pinecone: Whether to upload embeddings to Pinecone
            use_object_detection: Whether to use object detection + captioning pipeline
            
        Returns:
            Processing statistics and results
        """
        start_time = time.time()
        
        # Validate video file
        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")
        
        # Set video name
        if not video_name:
            video_name = Path(video_path).stem
        
        # Set video date (use today's date if not provided)
        if not video_date:
            from datetime import date
            video_date = date.today().strftime("%Y-%m-%d")
            logger.info(f"No video_date provided, using today's date: {video_date}")
        
        self.current_video = video_name
        logger.info(f"Processing video: {video_name} ({video_path})")
        logger.info(f"Video date: {video_date}")
        
        # Initialize components
        self._initialize_components()
        
        try:
            # Step 1: Extract frames
            logger.info("Step 1/4: Extracting frames...")
            frames = self.frame_extractor.extract_frames(
                video_path=video_path,
                use_similarity_filter=True,
                dedupe_method='clip' if getattr(self.config, 'ENABLE_CLIP_DEDUPE', False) else 'hist',
                video_date=video_date  # Pass video date to frame extractor
            )
            
            if save_frames:
                output_dir = os.path.join(self.config.OUTPUT_DIR, video_name, "frames")
                self.frame_extractor.save_frames_to_disk(output_dir)
            else:
                # Optionally save thumbnails even if full frames are not requested
                try:
                    if getattr(self.config, 'SAVE_THUMBNAILS', False):
                        thumb_dir = os.path.join(self.config.TEMP_DIR, video_name, 'frames')
                        os.makedirs(thumb_dir, exist_ok=True)
                        self.frame_extractor.save_frames_to_disk(thumb_dir)
                except Exception:
                    logger.debug("Thumbnail generation failed; continuing without thumbnails")
            
            # Enforce object detection mode irrespective of caller input
            use_object_detection = True

            # Step 2: Generate captions
            logger.info("Step 2/4: Generating captions...")
            
            if use_object_detection:
                # Use object-focused captioning pipeline
                logger.info("Using object detection + captioning pipeline")
                print("\n" + "="*80)
                print("🎯 OBJECT DETECTION + CAPTIONING MODE")
                print("="*80)
                
                if not self.object_pipeline:
                    from object_caption_pipeline import ObjectCaptionPipeline
                    self.object_pipeline = ObjectCaptionPipeline(
                        use_gpu=self.config.USE_GPU,
                        min_object_size=20,
                        max_objects_per_frame=10,
                        include_scene_caption=True,  # Ensure a scene caption even if no objects found
                        caption_similarity_threshold=0.85  # Filter similar captions
                    )
                
                # Reset caption history for new video
                self.object_pipeline.reset_caption_history()
                
                # Process frames with object detection (with verbose logging)
                object_captions = self.object_pipeline.process_frames(
                    frames=frames,
                    show_progress=True
                )
                
                # Convert ObjectCaption to CaptionedFrame format
                captioned_frames = []
                for oc in object_captions:
                    cf = CaptionedFrame(
                        frame_data=oc.frame_data,
                        caption=oc.attribute_caption,
                        confidence=oc.confidence
                    )
                    # Store namespace info in frame_data for later use
                    cf.frame_data.namespace = oc.namespace
                    captioned_frames.append(cf)

                logger.info(f"Object detection pipeline generated {len(captioned_frames)} captions")

                # Fallback: If object pipeline produced zero captions, use standard BLIP captioning
                if not captioned_frames:
                    logger.warning("Object detection produced no captions; falling back to standard BLIP captioning for scene-level captions.")
                    captioned_frames = self.caption_generator.generate_captions(
                        frames=frames,
                        filter_empty=True
                    )
                
            else:
                # Use standard BLIP captioning
                captioned_frames = self.caption_generator.generate_captions(
                    frames=frames,
                    filter_empty=True
                )
            
            # Fail fast if no captions were produced
            if not captioned_frames:
                raise ValueError("No captions were generated for any frames. Captioning is compulsory; check BLIP configuration or input video.")

            # Filter duplicate captions
            captioned_frames = self.caption_generator.filter_duplicate_captions(
                captioned_frames=captioned_frames,
                time_window=self.config.DUPLICATE_TIME_WINDOW
            )
            
            # Step 3: Generate embeddings
            logger.info("Step 3/4: Generating embeddings...")
            # If multimodal embeddings enabled, prefer MultimodalEmbeddingGenerator
            if getattr(self.config, 'ENABLE_DUAL_EMBEDDINGS', False) and hasattr(self.embedding_generator, 'generate_dual_embeddings'):
                # Some code paths may already swap in Multimodal generator; otherwise the TextEmbeddingGenerator will be used
                try:
                    embedded_frames = self.embedding_generator.generate_dual_embeddings(captioned_frames)
                except Exception:
                    embedded_frames = self.embedding_generator.generate_embeddings(captioned_frames)
            else:
                embedded_frames = self.embedding_generator.generate_embeddings(
                    captioned_frames=captioned_frames
                )
            
            # Step 3.5: Deduplicate embeddings before upload
            logger.info("Deduplicating embeddings...")
            captions_before_dedupe = len(embedded_frames)
            # Fallback if embedding generator lacks deduplication API
            if hasattr(self.embedding_generator, 'deduplicate_embeddings'):
                embedded_frames = self.embedding_generator.deduplicate_embeddings(
                    embedded_frames=embedded_frames,
                    similarity_threshold=0.95  # Remove very similar embeddings
                )
            else:
                embedded_frames = self._deduplicate_embeddings(
                    embedded_frames=embedded_frames,
                    similarity_threshold=0.95
                )
            logger.info(f"After deduplication: {len(embedded_frames)} unique embeddings")

            # Validate embeddings presence
            if getattr(self.config, 'ENABLE_DUAL_EMBEDDINGS', False):
                # When dual embeddings are enabled, ensure each embedded frame has both caption and image embeddings
                missing = [
                    ef.captioned_frame.frame_data.frame_id for ef in embedded_frames
                    if getattr(ef, 'caption_embedding', None) is None or getattr(ef, 'image_embedding', None) is None
                ]
                if missing:
                    raise ValueError(f"Dual embeddings are enabled but some frames lack caption/image embeddings: {missing[:5]}{'...' if len(missing) > 5 else ''}")

            # Compute temporal confidence scores (optional)
            if getattr(self.config, 'ENABLE_TEMPORAL_BOOTSTRAPPING', False):
                try:
                    tb = TemporalBootstrapper(
                        temporal_window=5,
                        confidence_threshold=getattr(self.config, 'CONFIDENCE_THRESHOLD', 0.5),
                        consistency_weight=0.3,
                        smoothing_sigma=1.5
                    )

                    embeddings_array = np.array([ef.embedding for ef in embedded_frames]) if embedded_frames else np.array([])
                    caption_confidences = [
                        (ef.captioned_frame.confidence if ef.captioned_frame.confidence is not None else 0.8)
                        for ef in embedded_frames
                    ]
                    frame_ids = [ef.captioned_frame.frame_data.frame_id for ef in embedded_frames]

                    if len(embeddings_array) > 0:
                        conf_scores = tb.propagate_confidence_scores(
                            embeddings=embeddings_array,
                            caption_confidences=caption_confidences,
                            frame_ids=frame_ids
                        )
                        # Annotate embedded_frames with combined confidence
                        for ef, cs in zip(embedded_frames, conf_scores):
                            ef.embedding_confidence = cs.combined_score

                        # Store mapping for later use
                        self.confidence_scores = {ef.captioned_frame.frame_data.frame_id: cs for ef, cs in zip(embedded_frames, conf_scores)}
                        logger.info("Temporal bootstrapping applied and confidence scores stored on engine")
                except Exception as e:
                    logger.warning(f"Temporal bootstrapping failed: {e}")
            
            # Step 4: Upload to Pinecone (combined + optional caption/image indices)
            actual_uploaded = 0
            if upload_to_pinecone:
                logger.info("Step 4/4: Uploading to Pinecone...")
                print("\n" + "="*80)
                print("☁️  UPLOADING TO PINECONE VECTOR DATABASE")
                print("="*80)
                
                pinecone_payload = self.embedding_generator.prepare_for_pinecone(
                    embedded_frames=embedded_frames,
                    video_name=video_name,
                    source_file_path=video_path
                )

                # Combined vectors (main index)
                pinecone_data = pinecone_payload.get('combined', [])
                caption_vectors = pinecone_payload.get('caption', [])
                image_vectors = pinecone_payload.get('image', [])
                
                # Group by namespace if using object detection
                if use_object_detection:
                    namespace_groups = {}
                    for i, (vec_id, vector, metadata) in enumerate(pinecone_data):
                        # Get namespace from metadata (which we just added in embedding_generator)
                        object_category = metadata.get('namespace', '')
                        
                        # Create date-based namespace: videos:YYYY-MM-DD:category
                        if object_category:
                            namespace = f"videos:{video_date}:{object_category}"
                        else:
                            namespace = f"videos:{video_date}:general"
                        
                        if namespace not in namespace_groups:
                            namespace_groups[namespace] = []
                        namespace_groups[namespace].append((vec_id, vector, metadata))
                    
                    # Upload each namespace separately with logging
                    for namespace, data in namespace_groups.items():
                        print(f"\n📁 Namespace: {namespace}")
                        print(f"   Uploading {len(data)} vectors...")
                        
                        uploaded = self.pinecone_manager.upload_embeddings(
                            data=data,
                            batch_size=self.config.PINECONE_BATCH_SIZE,
                            namespace=namespace
                        )
                        actual_uploaded += uploaded
                        
                        # Show sample
                        if data:
                            sample_caption = data[0][2].get('caption', 'N/A')
                            print(f"   ✓ Uploaded {uploaded} vectors")
                            print(f"   Sample caption: {sample_caption[:70]}...")
                else:
                    # Upload to default namespace (combined/main index)
                    actual_uploaded = self.pinecone_manager.upload_embeddings(
                        data=pinecone_data,
                        batch_size=self.config.PINECONE_BATCH_SIZE
                    )

                # If configured, optionally upload caption/image embeddings to separate indices
                try:
                    enable_dual = getattr(self.config, 'ENABLE_DUAL_EMBEDDINGS', False)
                    upload_separate = getattr(self.config, 'UPLOAD_SEPARATE_MODALITY_INDICES', False)
                except Exception:
                    enable_dual = False
                    upload_separate = False

                if enable_dual and upload_separate and caption_vectors:
                    text_index = getattr(self.config, 'PINECONE_TEXT_INDEX_NAME', self.config.PINECONE_INDEX_NAME)
                    text_dim = getattr(self.config, 'PINECONE_DIMENSION', self.config.PINECONE_DIMENSION)
                    uploaded_text = self.pinecone_manager.upload_embeddings_to_index(
                        index_name=text_index,
                        dimension=text_dim,
                        data=caption_vectors,
                        batch_size=self.config.PINECONE_BATCH_SIZE
                    )
                    logger.info(f"Uploaded {uploaded_text} caption vectors to index '{text_index}'")

                if enable_dual and upload_separate and image_vectors:
                    image_index = getattr(self.config, 'PINECONE_IMAGE_INDEX_NAME', None) or self.config.PINECONE_IMAGE_INDEX_NAME
                    image_dim = getattr(self.config, 'PINECONE_IMAGE_DIMENSION', self.config.PINECONE_IMAGE_DIMENSION)
                    uploaded_image = self.pinecone_manager.upload_embeddings_to_index(
                        index_name=image_index,
                        dimension=image_dim,
                        data=image_vectors,
                        batch_size=self.config.PINECONE_BATCH_SIZE
                    )
                    logger.info(f"Uploaded {uploaded_image} image vectors to index '{image_index}'")
                
                # Print verification
                print(f"\n{'='*80}")
                if actual_uploaded > 0:
                    print(f"✅ UPLOAD COMPLETE: {actual_uploaded} vectors uploaded for '{video_name}'")
                    sample_ids = [pinecone_data[i][0] for i in range(min(3, len(pinecone_data)))]
                    print(f"   Sample vector IDs: {', '.join(sample_ids[:3])}...")
                else:
                    print("❌ WARNING: No vectors were successfully uploaded to Pinecone")
                print(f"{'='*80}\n")
            
            # Store processed frames
            self.processed_frames = embedded_frames
            
            # Calculate statistics
            processing_time = time.time() - start_time
            
            # Calculate frame reduction correctly
            total_video_frames = len(frames)  # Frames before similarity filtering
            frames_after_caption = len(captioned_frames)  # Frames that got captions
            frame_reduction_pct = ((total_video_frames - frames_after_caption) / total_video_frames * 100) if total_video_frames > 0 else 0
            
            stats = {
                "video_name": video_name,
                "video_path": video_path,
                "total_frames_extracted": total_video_frames,
                "frames_with_captions": frames_after_caption,
                "captions_before_dedupe": captions_before_dedupe,
                "embeddings_generated": len(embedded_frames),  # After dedupe
                "embeddings_uploaded": actual_uploaded if upload_to_pinecone else 0,
                "processing_time_seconds": processing_time,
                "frame_reduction_percent": frame_reduction_pct,
                "caption_stats": self.caption_generator.get_caption_statistics(captioned_frames),
                "embedding_stats": self.embedding_generator.get_embedding_statistics(embedded_frames)
            }
            
            self.processing_stats = stats
            
            logger.info(f"Video processing completed in {processing_time:.2f} seconds")
            logger.info(f"Statistics: {json.dumps(stats, indent=2)}")
            
            # Save processing report
            self._save_processing_report(stats, video_name)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            raise
        
        finally:
            # Clear GPU cache
            if self.caption_generator:
                self.caption_generator.clear_gpu_cache()
            if self.embedding_generator:
                self.embedding_generator.clear_cache()
            if self.object_pipeline:
                self.object_pipeline.clear_cache()
    
    def _deduplicate_embeddings(self,
                               embedded_frames: List[EmbeddedFrame],
                               similarity_threshold: float = 0.95) -> List[EmbeddedFrame]:
        """Deduplicate embeddings locally if generator lacks the method."""
        if not embedded_frames:
            return []
        if len(embedded_frames) == 1:
            return embedded_frames
        # Stack embeddings
        embeddings = np.array([ef.embedding for ef in embedded_frames])
        # If embeddings are normalized, cosine similarity is dot product
        normalized = True
        # Heuristic: check mean norm ~1.0
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms.mean(), 1.0, atol=1e-2):
            normalized = False
        keep = np.ones(len(embeddings), dtype=bool)
        for i in range(len(embeddings)):
            if not keep[i]:
                continue
            vec_i = embeddings[i]
            for j in range(i + 1, len(embeddings)):
                if not keep[j]:
                    continue
                vec_j = embeddings[j]
                if normalized:
                    sim = float(np.dot(vec_i, vec_j))
                else:
                    denom = (np.linalg.norm(vec_i) * np.linalg.norm(vec_j)) or 1e-8
                    sim = float(np.dot(vec_i, vec_j) / denom)
                if sim >= similarity_threshold:
                    keep[j] = False
        return [ef for ef, k in zip(embedded_frames, keep) if k]

    def search(self,
              query: str,
              top_k: int = None,
              similarity_threshold: float = None,
              video_filter: Optional[str] = None,
              time_window: Optional[Tuple[float, float]] = None,
              date_filter: Optional[str] = None,
              namespace_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for video frames using natural language query
        
        Args:
            query: Natural language search query
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            video_filter: Filter by video name
            time_window: Filter by time range (start, end) in seconds
            date_filter: Filter by video date (YYYY-MM-DD format)
            namespace_filter: Filter by specific namespace/category
            
        Returns:
            List of search results with timestamps and metadata
        """
        # Use config defaults if not specified
        top_k = top_k or self.config.QUERY_TOP_K
        similarity_threshold = similarity_threshold or self.config.QUERY_SIMILARITY_THRESHOLD
        
        # Initialize components if needed
        if not self.embedding_generator:
            self.embedding_generator = TextEmbeddingGenerator(
                model_name=self.config.EMBEDDING_MODEL,
                batch_size=self.config.EMBEDDING_BATCH_SIZE,
                use_gpu=self.config.USE_GPU,
                normalize=True
            )
        
        if not self.pinecone_manager:
            self.pinecone_manager = PineconeManager(
                api_key=self.config.PINECONE_API_KEY,
                environment=self.config.PINECONE_ENVIRONMENT,
                index_name=self.config.PINECONE_INDEX_NAME,
                dimension=self.config.PINECONE_DIMENSION,
                metric=self.config.PINECONE_METRIC,
                host=getattr(self.config, 'PINECONE_HOST', None)
            )
        
        logger.info(f"Searching for: '{query}'")
        if date_filter:
            logger.info(f"Date filter: {date_filter}")
        if namespace_filter:
            logger.info(f"Namespace filter: {namespace_filter}")
        
        try:
            # Generate query embedding(s)
            query_embedding = self.embedding_generator.encode_query(query)
            # If dual embeddings enabled, also encode a CLIP text embedding for image index queries
            clip_query_embedding = None
            if getattr(self.config, 'ENABLE_DUAL_EMBEDDINGS', False):
                # Try to load a CLIP/text-image model for image-index queries with fallbacks
                clip_query_embedding = None
                tried = []
                for candidate in [getattr(self.config, 'CLIP_MODEL_NAME', None), 'clip-ViT-B-32', 'openai/clip-vit-base-patch32']:
                    if not candidate:
                        continue
                    try:
                        from sentence_transformers import SentenceTransformer
                        clip_model = SentenceTransformer(candidate, device='cpu')
                        clip_query_embedding = clip_model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
                        logger.info(f"Encoded query with CLIP model: {candidate}")
                        break
                    except Exception as e:
                        tried.append((candidate, str(e)))
                        logger.warning(f"Failed to load/encode with CLIP model '{candidate}': {e}")

                if clip_query_embedding is None:
                    logger.warning(f"Could not create CLIP query embedding. Tried: {[c[0] for c in tried]}")
            
            # If date_filter is provided, search date-specific namespaces
            if date_filter and namespace_filter:
                # Search specific date + category namespace
                target_namespace = f"videos:{date_filter}:{namespace_filter}"
                search_results = self.pinecone_manager.query(
                    query_vector=query_embedding,
                    top_k=top_k,
                    namespace=target_namespace,
                    include_metadata=True
                )
            elif date_filter:
                # Search all categories for this date
                # Get all namespaces and filter by date prefix
                stats = self.pinecone_manager.get_index_stats()
                namespaces = stats.get('namespaces', {})
                
                date_namespaces = [ns for ns in namespaces.keys() if ns.startswith(f"videos:{date_filter}:")]
                
                if not date_namespaces:
                    logger.warning(f"No namespaces found for date: {date_filter}")
                    return []
                
                # Query each namespace and combine results
                all_results = []
                for ns in date_namespaces:
                    ns_results = self.pinecone_manager.query(
                        query_vector=query_embedding,
                        top_k=top_k,
                        namespace=ns,
                        include_metadata=True
                    )
                    all_results.extend(ns_results)
                
                # Sort by score and take top_k
                all_results.sort(key=lambda x: x.score, reverse=True)
                search_results = all_results[:top_k]
            elif namespace_filter:
                # Search specific category across all dates
                # This requires querying multiple date namespaces
                stats = self.pinecone_manager.get_index_stats()
                namespaces = stats.get('namespaces', {})
                
                category_namespaces = [ns for ns in namespaces.keys() if ns.endswith(f":{namespace_filter}")]
                
                if not category_namespaces:
                    logger.warning(f"No namespaces found for category: {namespace_filter}")
                    return []
                
                # Query each namespace and combine results
                all_results = []
                for ns in category_namespaces:
                    ns_results = self.pinecone_manager.query(
                        query_vector=query_embedding,
                        top_k=top_k,
                        namespace=ns,
                        include_metadata=True
                    )
                    all_results.extend(ns_results)
                
                # Sort by score and take top_k
                all_results.sort(key=lambda x: x.score, reverse=True)
                search_results = all_results[:top_k]
            else:
                # No filters - search across ALL namespaces
                stats = self.pinecone_manager.get_index_stats()
                namespaces = stats.get('namespaces', {})
                
                if not namespaces:
                    # No namespaces exist, try default namespace
                    logger.warning("No namespaces found in index, searching default namespace")
                    # If dual indices are enabled, perform fusion search
                    if getattr(self.config, 'ENABLE_DUAL_EMBEDDINGS', False) and clip_query_embedding is not None:
                        # Query text index
                        text_index = getattr(self.config, 'PINECONE_TEXT_INDEX_NAME', self.config.PINECONE_INDEX_NAME)
                        image_index = getattr(self.config, 'PINECONE_IMAGE_INDEX_NAME', None) or self.config.PINECONE_IMAGE_INDEX_NAME

                        text_results = self.pinecone_manager.query_index(
                            index_name=text_index,
                            query_vector=query_embedding,
                            top_k=top_k,
                            include_metadata=True
                        )

                        image_results = self.pinecone_manager.query_index(
                            index_name=image_index,
                            query_vector=clip_query_embedding,
                            top_k=top_k,
                            include_metadata=True
                        )

                        # Normalize scores per list (min-max)
                        def normalize_list(res_list):
                            if not res_list:
                                return {}
                            scores = [r.score for r in res_list]
                            min_s, max_s = min(scores), max(scores)
                            norm = {}
                            for r in res_list:
                                if max_s - min_s <= 1e-6:
                                    norm_score = 1.0
                                else:
                                    norm_score = (r.score - min_s) / (max_s - min_s)
                                norm[r.metadata.get('parent_combined_id', r.id)] = norm_score
                            return norm

                        text_norm = normalize_list(text_results)
                        image_norm = normalize_list(image_results)

                        # Fuse by parent_combined_id (preferred) otherwise by id
                        fused = {}
                        text_w = getattr(self.config, 'FUSION_TEXT_WEIGHT', 0.6)
                        image_w = getattr(self.config, 'FUSION_IMAGE_WEIGHT', 0.4)

                        # Collect all keys
                        keys = set(list(text_norm.keys()) + list(image_norm.keys()))
                        for k in keys:
                            t = text_norm.get(k, 0.0)
                            im = image_norm.get(k, 0.0)
                            score = text_w * t + image_w * im
                            fused[k] = score

                        # Convert fused dict to list of tuples sorted
                        fused_items = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]

                        # Build formatted results, retrieving metadata from either text_results or image_results
                        id_to_result = {r.metadata.get('parent_combined_id', r.id): r for r in (text_results + image_results)}
                        final_results = []
                        for combined_id, fused_score in fused_items:
                            r = id_to_result.get(combined_id)
                            if not r:
                                continue
                            # Apply confidence weighting if present
                            combined_conf = r.metadata.get('combined_confidence') or r.metadata.get('combined_confidence', None)
                            if combined_conf is None:
                                combined_conf = 1.0
                            # Apply threshold filtering
                            if combined_conf < getattr(self.config, 'CONFIDENCE_THRESHOLD', 0.5):
                                # Skip low confidence
                                continue
                            final_results.append((r, fused_score * float(combined_conf)))

                        # Format for return
                        search_results = []
                        for r, score in final_results:
                            search_results.append(r)

                    else:
                        search_results = self.pinecone_manager.semantic_search(
                            query_embedding=query_embedding,
                            top_k=top_k,
                            similarity_threshold=similarity_threshold,
                            video_filter=video_filter,
                            time_window=time_window
                        )
                else:
                    # Query each namespace and combine results
                    logger.info(f"Searching across {len(namespaces)} namespaces")
                    all_results = []
                    for ns in namespaces.keys():
                        ns_results = self.pinecone_manager.query(
                            query_vector=query_embedding,
                            top_k=top_k,
                            namespace=ns,
                            include_metadata=True
                        )
                        all_results.extend(ns_results)
                    
                    # Filter by similarity threshold
                    filtered_results = [r for r in all_results if r.score >= similarity_threshold]
                    
                    # Filter by video if specified
                    if video_filter:
                        filtered_results = [r for r in filtered_results if r.video_name == video_filter]
                    
                    # Filter by time window if specified
                    if time_window:
                        start_time, end_time = time_window
                        filtered_results = [r for r in filtered_results if start_time <= r.timestamp <= end_time]
                    
                    # Sort by score and take top_k
                    filtered_results.sort(key=lambda x: x.score, reverse=True)
                    search_results = filtered_results[:top_k]
            
            # Format results
            formatted_results = []
            for result in search_results:
                formatted_result = {
                    "timestamp": result.timestamp,
                    "caption": result.caption,
                    "similarity_score": result.score,
                    "frame_id": result.frame_id,
                    "video_name": result.video_name,
                    "time_formatted": self._format_timestamp(result.timestamp)
                }
                formatted_results.append(formatted_result)
            
            logger.info(f"Found {len(formatted_results)} results for query: '{query}'")
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def batch_search(self,
                    queries: List[str],
                    top_k: int = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Perform batch search for multiple queries
        
        Args:
            queries: List of search queries
            top_k: Number of results per query
            
        Returns:
            Dictionary mapping queries to results
        """
        results = {}
        
        for query in queries:
            try:
                results[query] = self.search(query, top_k=top_k)
            except Exception as e:
                logger.error(f"Failed to search for '{query}': {e}")
                results[query] = []
        
        return results
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format timestamp in readable format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"
        else:
            return f"{minutes:02d}:{secs:05.2f}"
    
    def _save_processing_report(self, stats: Dict, video_name: str):
        """Save processing report to file"""
        report_dir = os.path.join(self.config.OUTPUT_DIR, video_name)
        os.makedirs(report_dir, exist_ok=True)
        
        report_path = os.path.join(report_dir, "processing_report.json")
        
        with open(report_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Processing report saved to: {report_path}")
    
    def get_index_stats(self) -> Dict:
        """Get Pinecone index statistics"""
        if not self.pinecone_manager:
            self.pinecone_manager = PineconeManager(
                api_key=self.config.PINECONE_API_KEY,
                environment=self.config.PINECONE_ENVIRONMENT,
                index_name=self.config.PINECONE_INDEX_NAME,
                dimension=self.config.PINECONE_DIMENSION,
                metric=self.config.PINECONE_METRIC,
                host=getattr(self.config, 'PINECONE_HOST', None)
            )
        
        return self.pinecone_manager.get_index_stats()
    
    def get_available_dates(self) -> List[str]:
        """
        Get list of all dates that have videos in the index
        
        Returns:
            List of dates in YYYY-MM-DD format, sorted
        """
        if not self.pinecone_manager:
            self.pinecone_manager = PineconeManager(
                api_key=self.config.PINECONE_API_KEY,
                environment=self.config.PINECONE_ENVIRONMENT,
                index_name=self.config.PINECONE_INDEX_NAME,
                dimension=self.config.PINECONE_DIMENSION,
                metric=self.config.PINECONE_METRIC,
                host=getattr(self.config, 'PINECONE_HOST', None)
            )
        
        stats = self.pinecone_manager.get_index_stats()
        namespaces = stats.get('namespaces', {})
        
        # Extract unique dates from namespace names
        dates = set()
        for ns in namespaces.keys():
            if ns.startswith('videos:'):
                parts = ns.split(':')
                if len(parts) >= 2:
                    date_part = parts[1]
                    # Validate date format (YYYY-MM-DD)
                    if len(date_part) == 10 and date_part[4] == '-' and date_part[7] == '-':
                        dates.add(date_part)
        
        return sorted(list(dates))
    
    def search_by_date_range(self,
                           query: str,
                           start_date: str,
                           end_date: str,
                           top_k: int = None,
                           similarity_threshold: float = None,
                           namespace_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search across a date range
        
        Args:
            query: Natural language search query
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            namespace_filter: Filter by specific category
            
        Returns:
            List of search results sorted by score
        """
        from datetime import datetime, timedelta
        
        top_k = top_k or self.config.QUERY_TOP_K
        similarity_threshold = similarity_threshold or self.config.QUERY_SIMILARITY_THRESHOLD
        
        # Generate date range
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        date_range = []
        current_dt = start_dt
        while current_dt <= end_dt:
            date_range.append(current_dt.strftime("%Y-%m-%d"))
            current_dt += timedelta(days=1)
        
        logger.info(f"Searching across date range: {start_date} to {end_date} ({len(date_range)} days)")
        
        # Initialize components
        if not self.embedding_generator:
            self.embedding_generator = TextEmbeddingGenerator(
                model_name=self.config.EMBEDDING_MODEL,
                batch_size=self.config.EMBEDDING_BATCH_SIZE,
                use_gpu=self.config.USE_GPU,
                normalize=True
            )
        
        if not self.pinecone_manager:
            self.pinecone_manager = PineconeManager(
                api_key=self.config.PINECONE_API_KEY,
                environment=self.config.PINECONE_ENVIRONMENT,
                index_name=self.config.PINECONE_INDEX_NAME,
                dimension=self.config.PINECONE_DIMENSION,
                metric=self.config.PINECONE_METRIC,
                host=getattr(self.config, 'PINECONE_HOST', None)
            )
        
        # Generate query embedding once
        query_embedding = self.embedding_generator.encode_query(query)
        
        # Get all namespaces
        stats = self.pinecone_manager.get_index_stats()
        all_namespaces = stats.get('namespaces', {}).keys()
        
        # Find relevant namespaces for date range
        target_namespaces = []
        for date in date_range:
            if namespace_filter:
                # Specific category
                ns = f"videos:{date}:{namespace_filter}"
                if ns in all_namespaces:
                    target_namespaces.append(ns)
            else:
                # All categories for this date
                date_ns = [ns for ns in all_namespaces if ns.startswith(f"videos:{date}:")]
                target_namespaces.extend(date_ns)
        
        if not target_namespaces:
            logger.warning(f"No namespaces found for date range {start_date} to {end_date}")
            return []
        
        logger.info(f"Searching {len(target_namespaces)} namespaces")
        
        # Query each namespace and collect results
        all_results = []
        for ns in target_namespaces:
            ns_results = self.pinecone_manager.query(
                query_vector=query_embedding,
                top_k=top_k,
                namespace=ns,
                include_metadata=True
            )
            all_results.extend(ns_results)
        
        # Filter by similarity threshold
        filtered_results = [r for r in all_results if r.score >= similarity_threshold]
        
        # Sort by score and take top_k
        filtered_results.sort(key=lambda x: x.score, reverse=True)
        top_results = filtered_results[:top_k]
        
        # Format results
        formatted_results = []
        for result in top_results:
            formatted_result = {
                "timestamp": result.timestamp,
                "caption": result.caption,
                "similarity_score": result.score,
                "frame_id": result.frame_id,
                "video_name": result.video_name,
                "video_date": result.metadata.get('video_date', 'unknown'),
                "time_formatted": self._format_timestamp(result.timestamp)
            }
            formatted_results.append(formatted_result)
        
        logger.info(f"Found {len(formatted_results)} results across date range")
        
        return formatted_results
    
    def clear_index(self) -> bool:
        """Clear all vectors from Pinecone index"""
        if not self.pinecone_manager:
            self.pinecone_manager = PineconeManager(
                api_key=self.config.PINECONE_API_KEY,
                environment=self.config.PINECONE_ENVIRONMENT,
                index_name=self.config.PINECONE_INDEX_NAME,
                dimension=self.config.PINECONE_DIMENSION,
                metric=self.config.PINECONE_METRIC,
                host=getattr(self.config, 'PINECONE_HOST', None)
            )
        
        return self.pinecone_manager.clear_index()
    
    def cleanup(self):
        """Clean up resources"""
        if self.caption_generator:
            self.caption_generator.unload_model()
        if self.embedding_generator:
            self.embedding_generator.unload_model()
        if self.object_pipeline:
            self.object_pipeline.unload_models()
        
        logger.info("Resources cleaned up")


def demo_usage():
    """Demonstrate usage of the Video Search Engine"""
    
    # Initialize engine
    engine = VideoSearchEngine()
    
    # Example: Process a video
    # stats = engine.process_video(
    #     video_path="sample_video.mp4",
    #     video_name="sample_demo",
    #     save_frames=False,
    #     upload_to_pinecone=True
    # )
    
    # Example: Search for content
    # results = engine.search(
    #     query="person walking with a black bag",
    #     top_k=5
    # )
    # 
    # for result in results:
    #     print(f"Time: {result['time_formatted']} - Score: {result['similarity_score']:.3f}")
    #     print(f"  Caption: {result['caption']}")
    #     print(f"  Video: {result['video_name']}")
    #     print()
    
    # Example: Batch search
    # queries = [
    #     "black bag",
    #     "yellow bottle",
    #     "person walking",
    #     "car driving"
    # ]
    # 
    # batch_results = engine.batch_search(queries, top_k=3)
    # 
    # for query, results in batch_results.items():
    #     print(f"\nQuery: '{query}' - Found {len(results)} results")
    #     for result in results[:2]:  # Show top 2
    #         print(f"  {result['time_formatted']} (score: {result['similarity_score']:.3f})")
    
    # Get index stats
    stats = engine.get_index_stats()
    print(f"Index statistics: {json.dumps(stats, indent=2)}")
    
    # Cleanup
    engine.cleanup()


if __name__ == "__main__":
    demo_usage()