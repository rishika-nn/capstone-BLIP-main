"""
Configuration file for Video Frame Search System
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the video search system"""
    
    # API Keys (load from environment variables for security)
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY', '')
    PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT', 'us-east-1')
    PINECONE_HOST = os.getenv('PINECONE_HOST', 'https://test-xp69mf0.svc.aped-4627-b74a.pinecone.io')
    
    # Pinecone Index Configuration
    PINECONE_INDEX_NAME = 'test'
    # Main index dimension (combined vector). With dual embeddings (text 1024 + image 512),
    # the combined embedding is padded to the larger dimension (1024) to match existing index configs.
    PINECONE_DIMENSION = 1024
    PINECONE_METRIC = 'cosine'
    PINECONE_CLOUD = 'aws'
    PINECONE_REGION = 'us-east-1'
    # Optional separate indices for multi-modal storage
    PINECONE_TEXT_INDEX_NAME = 'test-text'
    PINECONE_IMAGE_INDEX_NAME = 'test-image'
    PINECONE_IMAGE_DIMENSION = 512  # Typical CLIP image embedding dim
    
    # Model Configuration
    # Use larger BLIP for better descriptive captions (requires more VRAM)
    BLIP_MODEL = 'Salesforce/blip-image-captioning-large'
    # Using multilingual-e5-large for better semantic understanding
    EMBEDDING_MODEL = 'intfloat/multilingual-e5-large'  # 1024 dimensions
    # Alternative: 'sentence-transformers/all-MiniLM-L6-v2' (384 dimensions)
    
    # Frame Extraction Configuration
    FRAME_SIMILARITY_THRESHOLD = 0.90  # Higher threshold to capture more frames (only skip very similar frames)
    FRAME_EXTRACTION_INTERVAL = 2.0  # Extract frame every N seconds (if not using similarity)
    MAX_FRAMES_PER_VIDEO = 1000  # Maximum frames to extract per video
    FRAME_RESIZE_WIDTH = 640  # Resize frames for memory efficiency (None for original size)
    MIN_FRAMES_PER_VIDEO = 10  # Minimum frames to extract regardless of similarity
    
    # Enhanced Caption Configuration
    GENERATE_MULTIPLE_CAPTIONS = True  # Generate multiple object-focused captions per frame
    CAPTIONS_PER_FRAME = 5  # Try more variations per frame for better recall
    USE_OBJECT_FOCUSED_PROMPTS = True  # Use object-focused prompts for more detailed descriptions
    # Use CLIP-based reranking to pick the best caption per frame
    ENABLE_CLIP_RERANK = True
    # Optionally override rerank model; if None, fall back to CLIP_MODEL_NAME
    CLIP_RERANK_MODEL = None
    
    # Processing Configuration
    BLIP_BATCH_SIZE = 8  # Batch size for BLIP caption generation
    EMBEDDING_BATCH_SIZE = 32  # Batch size for embedding generation
    PINECONE_BATCH_SIZE = 100  # Batch size for Pinecone uploads
    # Enhanced options
    ENABLE_DUAL_EMBEDDINGS = True  # Compute caption + image embeddings and combined vector
    UPLOAD_SEPARATE_MODALITY_INDICES = False  # If True, upload caption/image vectors to separate indices (may hit index limits)
    ENABLE_TEMPORAL_BOOTSTRAPPING = True  # Compute temporal confidence and smoothing
    CONFIDENCE_THRESHOLD = 0.5  # Default threshold for filtering search results
    TEMPORAL_WINDOW = 7
    SMOOTHING_SIGMA = 2.0
    # Fusion / multi-index search options
    FUSION_TEXT_WEIGHT = 0.7  # Text tends to be more reliable for classroom/surveillance
    FUSION_IMAGE_WEIGHT = 0.3
    # Preferred CLIP model identifier(s). Use a valid SentenceTransformers/CLIP model id.
    # Common working IDs: 'clip-ViT-B-32' or 'openai/clip-vit-base-patch32'
    CLIP_MODEL_NAME = 'clip-ViT-B-32'  # CLIP model for image embeddings and text->image queries
    ENABLE_CLIP_DEDUPE = False  # If True, use CLIP-based semantic dedupe instead of histogram-based
    CLIP_DEDUPE_THRESHOLD = 0.88  # Similarity threshold for CLIP semantic dedupe (0-1)

    # Frame extraction
    FRAME_RESIZE_WIDTH = 960  # Process higher-res frames for small-object detail
    # Thumbnails
    SAVE_THUMBNAILS = True  # Save small thumbnails for UI and metadata
    THUMBNAIL_SIZE = (256, 256)  # Size of generated thumbnails (width, height)
    
    # Query Configuration
    QUERY_TOP_K = 10  # Number of results to return
    QUERY_SIMILARITY_THRESHOLD = 0.6  # Minimum similarity score for results
    DUPLICATE_TIME_WINDOW = 5.0  # Seconds within which to consider frames as duplicates (increased for multi-captions)
    
    # Logging Configuration
    LOG_LEVEL = 'INFO'
    LOG_FILE = 'video_search.log'
    
    # Performance Configuration
    USE_GPU = True  # Use GPU if available
    NUM_WORKERS = 4  # Number of workers for data loading
    
    # File paths
    TEMP_DIR = './temp'
    OUTPUT_DIR = './output'
    
    @classmethod
    def validate(cls):
        """Validate configuration settings"""
        if not cls.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not set. Please set it in .env file or environment variables")
        
        if cls.PINECONE_DIMENSION not in [384, 768, 1024, 1536]:
            print(f"Warning: Unusual embedding dimension {cls.PINECONE_DIMENSION}. Common values are 384, 768, 1024, or 1536")
        
        if cls.FRAME_SIMILARITY_THRESHOLD < 0 or cls.FRAME_SIMILARITY_THRESHOLD > 1:
            raise ValueError("FRAME_SIMILARITY_THRESHOLD must be between 0 and 1")
        
        # Create directories if they don't exist
        os.makedirs(cls.TEMP_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        
        return True