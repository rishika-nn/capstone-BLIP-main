"""
Text Embedding Generation Module
Converts captions into dense vector embeddings for semantic search
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import logging
from typing import List, Optional, Union, Tuple, Dict
from tqdm import tqdm
import gc
from dataclasses import dataclass
from caption_generator import CaptionedFrame

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EmbeddedFrame:
    """Data structure for frame with caption and embedding"""
    captioned_frame: CaptionedFrame
    embedding: np.ndarray
    embedding_id: str
    image_embedding: Optional[np.ndarray] = None  # Image visual embedding
    image_embedding_id: Optional[str] = None  # ID for image embedding
    caption_embedding: Optional[np.ndarray] = None  # Caption text embedding
    caption_embedding_id: Optional[str] = None  # ID for caption embedding
    embedding_confidence: Optional[float] = None  # Quality confidence for embedding

class TextEmbeddingGenerator:
    """Generate embeddings for text using sentence-transformers"""
    
    def __init__(self,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 batch_size: int = 32,
                 use_gpu: bool = True,
                 normalize: bool = True):
        """
        Initialize the text embedding generator
        
        Args:
            model_name: Sentence transformer model name
            batch_size: Batch size for encoding
            use_gpu: Whether to use GPU if available
            normalize: Whether to normalize embeddings for cosine similarity
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        
        # Set device
        self.device = 'cuda' if torch.cuda.is_available() and use_gpu else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        # Load model
        self._load_model()
        
        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def _load_model(self):
        """Load sentence transformer model"""
        logger.info(f"Loading embedding model: {self.model_name}")
        
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            logger.info("Embedding model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def generate_embeddings(self, 
                          captioned_frames: List[CaptionedFrame],
                          show_progress: bool = True) -> List[EmbeddedFrame]:
        """
        Generate embeddings for captioned frames
        
        Args:
            captioned_frames: List of CaptionedFrame objects
            show_progress: Whether to show progress bar
            
        Returns:
            List of EmbeddedFrame objects
        """
        if not captioned_frames:
            return []
        
        logger.info(f"Generating embeddings for {len(captioned_frames)} captions")
        
        # Extract captions
        captions = [cf.caption for cf in captioned_frames]
        
        # Generate embeddings
        embeddings = self._encode_batch(captions, show_progress=show_progress)
        
        # Create EmbeddedFrame objects
        embedded_frames = []
        for cf, embedding in zip(captioned_frames, embeddings):
            # Generate unique embedding ID
            embedding_id = f"{cf.frame_data.frame_id}_emb"
            
            embedded_frame = EmbeddedFrame(
                captioned_frame=cf,
                embedding=embedding,
                embedding_id=embedding_id
            )
            embedded_frames.append(embedded_frame)
        
        logger.info(f"Generated {len(embedded_frames)} embeddings")
        return embedded_frames
    
    def _encode_batch(self, texts: List[str], 
                     show_progress: bool = True) -> np.ndarray:
        """
        Encode a batch of texts into embeddings
        
        Args:
            texts: List of text strings
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings [N, embedding_dim]
        """
        # Encode with sentence transformer
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize
        )
        
        return embeddings
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query text
        
        Args:
            query: Query string
            
        Returns:
            Query embedding vector
        """
        # Encode single query
        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize
        )
        
        # Ensure it's a 1D array
        if len(embedding.shape) > 1:
            embedding = embedding.squeeze()
        
        return embedding
    
    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """
        Encode multiple queries
        
        Args:
            queries: List of query strings
            
        Returns:
            Array of query embeddings [N, embedding_dim]
        """
        embeddings = self.model.encode(
            queries,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize
        )
        
        return embeddings
    
    def compute_similarity(self, 
                          query_embedding: np.ndarray,
                          embeddings: np.ndarray) -> np.ndarray:
        """
        Compute similarity between query and embeddings
        
        Args:
            query_embedding: Query embedding vector
            embeddings: Array of embeddings to compare against
            
        Returns:
            Similarity scores
        """
        if self.normalize:
            # If normalized, use dot product (equivalent to cosine similarity)
            similarities = np.dot(embeddings, query_embedding)
        else:
            # Compute cosine similarity manually
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            similarities = np.dot(embeddings_norm, query_norm)
        
        return similarities
    
    def find_similar(self,
                    query: str,
                    embedded_frames: List[EmbeddedFrame],
                    top_k: int = 10,
                    threshold: float = 0.5) -> List[Tuple[EmbeddedFrame, float]]:
        """
        Find similar frames to a query
        
        Args:
            query: Query string
            embedded_frames: List of EmbeddedFrame objects
            top_k: Number of top results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (EmbeddedFrame, similarity_score) tuples
        """
        # Encode query
        query_embedding = self.encode_query(query)
        
        # Extract embeddings from frames
        embeddings = np.array([ef.embedding for ef in embedded_frames])
        
        # Compute similarities
        similarities = self.compute_similarity(query_embedding, embeddings)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Filter by threshold and create results
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= threshold:
                results.append((embedded_frames[idx], score))
        
        return results
    
    def create_embedding_matrix(self, 
                               embedded_frames: List[EmbeddedFrame]) -> Tuple[np.ndarray, List[str]]:
        """
        Create embedding matrix and ID list for batch operations
        
        Args:
            embedded_frames: List of EmbeddedFrame objects
            
        Returns:
            Tuple of (embedding_matrix, embedding_ids)
        """
        embeddings = []
        ids = []
        
        for ef in embedded_frames:
            embeddings.append(ef.embedding)
            ids.append(ef.embedding_id)
        
        embedding_matrix = np.array(embeddings)
        
        return embedding_matrix, ids
    
    def prepare_for_pinecone(self, 
                           embedded_frames: List[EmbeddedFrame],
                           video_name: str = "video",
                           source_file_path: str = "") -> List[Tuple[str, List[float], Dict]]:
        """
        Prepare data for Pinecone upload
        
        Args:
            embedded_frames: List of EmbeddedFrame objects
            video_name: Name of the video for metadata
            
        Returns:
            List of (id, vector, metadata) tuples for Pinecone
        """
        """
        Prepare multi-index data for Pinecone upload.

        Returns a dict with keys: 'combined', 'caption', 'image' each mapping to
        a list of (id, vector, metadata) tuples.
        """
        combined_data = []
        caption_data = []
        image_data = []

        # Basic color list for lightweight attribute extraction
        color_words = set(['black','white','gray','grey','red','orange','yellow','green','blue','purple','pink','brown','beige','tan','gold','silver','navy','maroon','teal'])

        for idx, ef in enumerate(embedded_frames):
            frame_id = ef.captioned_frame.frame_data.frame_id
            timestamp = ef.captioned_frame.frame_data.timestamp

            # IDs
            combined_id = f"{frame_id}_obj{idx}_combined"
            caption_id = f"{frame_id}_obj{idx}_caption"
            image_id = f"{frame_id}_obj{idx}_image"

            # Combined vector (the main vector used for primary index)
            combined_vector = ef.embedding.tolist()

            # Caption and image vectors (if available)
            caption_vector = ef.caption_embedding.tolist() if ef.caption_embedding is not None else None
            image_vector = ef.image_embedding.tolist() if ef.image_embedding is not None else None

            # Extract simple structured attributes (colors/nouns) from caption text
            caption_text = ef.captioned_frame.caption or ''
            caption_lower = caption_text.lower()
            found_colors = [c for c in color_words if c in caption_lower]

            metadata = {
                'timestamp': timestamp,
                'caption': caption_text,
                'frame_id': frame_id,
                'frame_index': ef.captioned_frame.frame_data.frame_index,
                'video_name': video_name,
                'source_file_path': source_file_path,
                'video_date': ef.captioned_frame.frame_data.video_date,
                'namespace': getattr(ef.captioned_frame.frame_data, 'namespace', ''),
                'object_label': getattr(ef.captioned_frame.frame_data, 'object_label', ''),
                'colors': found_colors,
                # Embedding ids for traceability
                'caption_embedding_id': caption_id,
                'image_embedding_id': image_id,
                'combined_confidence': float(ef.embedding_confidence) if ef.embedding_confidence is not None else None,
                'thumbnail_path': getattr(ef.captioned_frame.frame_data, 'thumbnail_path', None)
            }

            # Append combined
            combined_data.append((combined_id, combined_vector, metadata))

            # Caption index entry
            if caption_vector is not None:
                caption_meta = metadata.copy()
                caption_meta.update({'parent_combined_id': combined_id, 'modality': 'caption'})
                caption_data.append((caption_id, caption_vector, caption_meta))

            # Image index entry
            if image_vector is not None:
                image_meta = metadata.copy()
                image_meta.update({'parent_combined_id': combined_id, 'modality': 'image'})
                image_data.append((image_id, image_vector, image_meta))

        return {
            'combined': combined_data,
            'caption': caption_data,
            'image': image_data
        }
    
    def augment_embeddings(self,
                          embedded_frames: List[EmbeddedFrame],
                          augmentation_factor: float = 0.1) -> List[EmbeddedFrame]:
        """
        Augment embeddings with small perturbations for better retrieval
        
        Args:
            embedded_frames: List of EmbeddedFrame objects
            augmentation_factor: Factor for perturbation (0.0 to 1.0)
            
        Returns:
            Augmented EmbeddedFrame objects
        """
        augmented = []
        
        for ef in embedded_frames:
            # Add small random perturbation
            noise = np.random.randn(*ef.embedding.shape) * augmentation_factor
            augmented_embedding = ef.embedding + noise
            
            # Re-normalize if needed
            if self.normalize:
                augmented_embedding = augmented_embedding / np.linalg.norm(augmented_embedding)
            
            # Create new EmbeddedFrame with augmented embedding
            augmented_ef = EmbeddedFrame(
                captioned_frame=ef.captioned_frame,
                embedding=augmented_embedding,
                embedding_id=f"{ef.embedding_id}_aug"
            )
            
            augmented.append(ef)  # Keep original
            augmented.append(augmented_ef)  # Add augmented
        
        return augmented
    
    def deduplicate_embeddings(self,
                              embedded_frames: List[EmbeddedFrame],
                              similarity_threshold: float = 0.95) -> List[EmbeddedFrame]:
        """
        Remove duplicate embeddings based on similarity threshold
        
        Args:
            embedded_frames: List of EmbeddedFrame objects
            similarity_threshold: Minimum similarity to consider as duplicate (0.0 to 1.0)
            
        Returns:
            List of unique EmbeddedFrame objects
        """
        if not embedded_frames:
            return []
        
        if len(embedded_frames) <= 1:
            return embedded_frames
        
        logger.info(f"Deduplicating {len(embedded_frames)} embeddings with threshold {similarity_threshold}")
        
        # Convert to numpy array for efficient computation
        embeddings = np.array([ef.embedding for ef in embedded_frames])
        
        # Track which embeddings to keep
        keep_mask = np.ones(len(embedded_frames), dtype=bool)
        
        # Compare each embedding with subsequent ones
        for i in range(len(embeddings)):
            if not keep_mask[i]:
                continue
            
            # Compute similarity with all subsequent embeddings
            for j in range(i + 1, len(embeddings)):
                if not keep_mask[j]:
                    continue
                
                # Compute cosine similarity
                if self.normalize:
                    # If normalized, use dot product
                    similarity = np.dot(embeddings[i], embeddings[j])
                else:
                    # Compute cosine similarity manually
                    similarity = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                
                # Mark as duplicate if similarity exceeds threshold
                if similarity >= similarity_threshold:
                    keep_mask[j] = False
        
        # Filter embeddings based on keep mask
        unique_frames = [ef for ef, keep in zip(embedded_frames, keep_mask) if keep]
        
        removed_count = len(embedded_frames) - len(unique_frames)
        logger.info(f"Removed {removed_count} duplicate embeddings, kept {len(unique_frames)} unique")
        
        return unique_frames
    
    def get_embedding_statistics(self, embedded_frames: List[EmbeddedFrame]) -> Dict:
        """Get statistics about embeddings"""
        if not embedded_frames:
            return {"total": 0}
        
        embeddings = np.array([ef.embedding for ef in embedded_frames])
        
        stats = {
            "total": len(embeddings),
            "dimension": embeddings.shape[1],
            "mean_norm": float(np.mean(np.linalg.norm(embeddings, axis=1))),
            "std_norm": float(np.std(np.linalg.norm(embeddings, axis=1))),
            "mean_similarity": float(np.mean(np.dot(embeddings, embeddings.T)))
        }
        
        return stats
    
    def clear_cache(self):
        """Clear cache and free memory"""
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        logger.info("Cache cleared")
    
    def unload_model(self):
        """Unload model from memory"""
        del self.model
        self.clear_cache()
        logger.info("Embedding model unloaded")


class MultimodalEmbeddingGenerator:
    """
    Generate both image and caption embeddings for improved multimodal search
    Stores caption embeddings and image embeddings separately for flexible retrieval
    """
    
    def __init__(self,
                 caption_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 image_model: str = "clip-ViT-B-32",
                 batch_size: int = 32,
                 use_gpu: bool = True,
                 normalize: bool = True,
                 embedding_weights: Dict[str, float] = None):
        """
        Initialize multimodal embedding generator
        
        Args:
            caption_model: Model for text embeddings
            image_model: Model for image embeddings (CLIP-based)
            batch_size: Batch size for encoding
            use_gpu: Whether to use GPU
            normalize: Whether to normalize embeddings
            embedding_weights: Weights for combining caption and image embeddings
                               {'caption': 0.6, 'image': 0.4} by default
        """
        self.batch_size = batch_size
        self.normalize = normalize
        self.device = 'cuda' if torch.cuda.is_available() and use_gpu else 'cpu'
        
        # Default weights
        if embedding_weights is None:
            embedding_weights = {'caption': 0.6, 'image': 0.4}
        self.embedding_weights = embedding_weights
        
        logger.info(f"Initializing multimodal embeddings on {self.device}")
        logger.info(f"Embedding weights - Caption: {embedding_weights['caption']}, "
                   f"Image: {embedding_weights['image']}")
        
        # Load caption model
        logger.info(f"Loading caption model: {caption_model}")
        self.caption_model = SentenceTransformer(caption_model, device=self.device)
        self.caption_model.eval()
        
        # Load image model
        logger.info(f"Loading image model: {image_model}")
        # Try common model ids with graceful fallback
        tried = []
        for candidate in [image_model, 'clip-ViT-B-32', 'openai/clip-vit-base-patch32']:
            try:
                self.image_model = SentenceTransformer(candidate, device=self.device)
                logger.info(f"Loaded image model: {candidate}")
                break
            except Exception as e:
                tried.append((candidate, str(e)))
                logger.warning(f"Failed to load image model '{candidate}': {e}")
        else:
            logger.error(f"Could not load any CLIP image model from candidates: {[c[0] for c in tried]}")
            raise RuntimeError("Failed to load CLIP image model; check model id and internet/authentication")
        self.image_model.eval()
        
        self.caption_dim = self.caption_model.get_sentence_embedding_dimension()
        self.image_dim = self.image_model.get_sentence_embedding_dimension()
        
        logger.info(f"Caption embedding dimension: {self.caption_dim}")
        logger.info(f"Image embedding dimension: {self.image_dim}")
    
    def generate_dual_embeddings(self,
                                captioned_frames: List[CaptionedFrame],
                                show_progress: bool = True) -> List[EmbeddedFrame]:
        """
        Generate both caption and image embeddings for all frames
        
        Args:
            captioned_frames: List of CaptionedFrame objects
            show_progress: Whether to show progress
            
        Returns:
            List of EmbeddedFrame objects with dual embeddings
        """
        if not captioned_frames:
            return []
        
        logger.info(f"Generating dual embeddings for {len(captioned_frames)} frames")
        
        embedded_frames = []
        
        for cf in tqdm(captioned_frames, desc="Dual embedding generation", 
                       disable=not show_progress):
            # Generate caption embedding
            caption_embedding = self.caption_model.encode(
                cf.caption,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize
            )
            
            # Generate image embedding
            image_embedding = self.image_model.encode(
                cf.frame_data.image,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize
            )
            
            # Combine embeddings with weights
            combined_embedding = self._combine_embeddings(
                caption_embedding,
                image_embedding
            )

            # Confidence: compute dot on aligned dimensions
            # Ensure same dimension for confidence computation
            if caption_embedding.shape[0] != image_embedding.shape[0]:
                target_dim = min(caption_embedding.shape[0], image_embedding.shape[0])
                cap_conf = caption_embedding[:target_dim]
                img_conf = image_embedding[:target_dim]
            else:
                cap_conf = caption_embedding
                img_conf = image_embedding
            alignment_confidence = float(np.dot(cap_conf, img_conf))
            
            embedded_frame = EmbeddedFrame(
                captioned_frame=cf,
                embedding=combined_embedding,
                embedding_id=f"{cf.frame_data.frame_id}_combined",
                caption_embedding=caption_embedding,
                caption_embedding_id=f"{cf.frame_data.frame_id}_caption",
                image_embedding=image_embedding,
                image_embedding_id=f"{cf.frame_data.frame_id}_image",
                embedding_confidence=alignment_confidence
            )
            embedded_frames.append(embedded_frame)
        
        logger.info(f"Generated {len(embedded_frames)} dual embeddings")
        return embedded_frames

    def generate_embeddings(self,
                            captioned_frames: List[CaptionedFrame],
                            show_progress: bool = True) -> List[EmbeddedFrame]:
        """
        Compatibility wrapper to provide the same API as TextEmbeddingGenerator.
        Generates dual embeddings and returns the combined embedding as primary.
        """
        return self.generate_dual_embeddings(captioned_frames, show_progress=show_progress)
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode a text query using the caption model (for search)."""
        embedding = self.caption_model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize
        )
        if hasattr(embedding, 'shape') and len(embedding.shape) > 1:
            embedding = embedding.squeeze()
        return embedding
    
    def prepare_for_pinecone(self,
                             embedded_frames: List[EmbeddedFrame],
                             video_name: str = "video",
                             source_file_path: str = "") -> Dict[str, List[Tuple[str, List[float], Dict]]]:
        """
        Prepare multi-index data for Pinecone upload.
        Returns a dict with keys: 'combined', 'caption', 'image' each mapping to
        a list of (id, vector, metadata) tuples.
        """
        combined_data = []
        caption_data = []
        image_data = []

        color_words = set(['black','white','gray','grey','red','orange','yellow','green','blue','purple','pink','brown','beige','tan','gold','silver','navy','maroon','teal'])

        for idx, ef in enumerate(embedded_frames):
            frame_id = ef.captioned_frame.frame_data.frame_id
            timestamp = ef.captioned_frame.frame_data.timestamp

            combined_id = f"{frame_id}_obj{idx}_combined"
            caption_id = f"{frame_id}_obj{idx}_caption"
            image_id = f"{frame_id}_obj{idx}_image"

            combined_vector = ef.embedding.tolist()
            caption_vector = ef.caption_embedding.tolist() if ef.caption_embedding is not None else None
            image_vector = ef.image_embedding.tolist() if ef.image_embedding is not None else None

            caption_text = ef.captioned_frame.caption or ''
            caption_lower = caption_text.lower()
            found_colors = [c for c in color_words if c in caption_lower]

            metadata = {
                'timestamp': timestamp,
                'caption': caption_text,
                'frame_id': frame_id,
                'frame_index': ef.captioned_frame.frame_data.frame_index,
                'video_name': video_name,
                'source_file_path': source_file_path,
                'video_date': ef.captioned_frame.frame_data.video_date,
                'namespace': getattr(ef.captioned_frame.frame_data, 'namespace', ''),
                'object_label': getattr(ef.captioned_frame.frame_data, 'object_label', ''),
                'colors': found_colors,
                'caption_embedding_id': caption_id,
                'image_embedding_id': image_id,
                'combined_confidence': float(ef.embedding_confidence) if ef.embedding_confidence is not None else None,
                'thumbnail_path': getattr(ef.captioned_frame.frame_data, 'thumbnail_path', None)
            }

            combined_data.append((combined_id, combined_vector, metadata))

            if caption_vector is not None:
                caption_meta = metadata.copy()
                caption_meta.update({'parent_combined_id': combined_id, 'modality': 'caption'})
                caption_data.append((caption_id, caption_vector, caption_meta))

            if image_vector is not None:
                image_meta = metadata.copy()
                image_meta.update({'parent_combined_id': combined_id, 'modality': 'image'})
                image_data.append((image_id, image_vector, image_meta))

        return {
            'combined': combined_data,
            'caption': caption_data,
            'image': image_data
        }
    
    def _combine_embeddings(self,
                           caption_emb: np.ndarray,
                           image_emb: np.ndarray) -> np.ndarray:
        """
        Combine caption and image embeddings with weights
        
        Args:
            caption_emb: Caption embedding
            image_emb: Image embedding
            
        Returns:
            Combined embedding
        """
        # Ensure same dimension by padding the smaller to the larger
        if caption_emb.shape[0] != image_emb.shape[0]:
            target_dim = max(caption_emb.shape[0], image_emb.shape[0])
            if caption_emb.shape[0] < target_dim:
                pad = np.zeros(target_dim - caption_emb.shape[0], dtype=caption_emb.dtype)
                caption_emb = np.concatenate([caption_emb, pad], axis=0)
            if image_emb.shape[0] < target_dim:
                pad = np.zeros(target_dim - image_emb.shape[0], dtype=image_emb.dtype)
                image_emb = np.concatenate([image_emb, pad], axis=0)
        
        # Weighted combination
        combined = (self.embedding_weights['caption'] * caption_emb +
                   self.embedding_weights['image'] * image_emb)
        
        # Normalize combined embedding
        if self.normalize:
            norm = np.linalg.norm(combined)
            if norm > 0:
                combined = combined / norm
        
        return combined
    
    def encode_query_multimodal(self,
                               text_query: str,
                               image_query: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Encode a query using multimodal approach
        
        Args:
            text_query: Text query string
            image_query: Optional image query (numpy array)
            
        Returns:
            Combined query embedding
        """
        # Encode text
        text_embedding = self.caption_model.encode(
            text_query,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize
        )
        
        if image_query is not None:
            # Encode image
            image_embedding = self.image_model.encode(
                image_query,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize
            )
            
            # Combine
            combined = self._combine_embeddings(text_embedding, image_embedding)
        else:
            # Only text query
            combined = text_embedding
        
        return combined
    
    def clear_cache(self):
        """Clear GPU cache"""
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        logger.info("Cache cleared")
    
    def unload_models(self):
        """Unload models from memory"""
        del self.caption_model
        del self.image_model
        self.clear_cache()
        logger.info("Multimodal models unloaded")

    def get_embedding_statistics(self, embedded_frames: List[EmbeddedFrame]) -> Dict:
        """Get statistics about combined embeddings for multimodal generator"""
        if not embedded_frames:
            return {"total": 0}
        embeddings = np.array([ef.embedding for ef in embedded_frames])
        stats = {
            "total": len(embeddings),
            "dimension": embeddings.shape[1],
            "mean_norm": float(np.mean(np.linalg.norm(embeddings, axis=1))),
            "std_norm": float(np.std(np.linalg.norm(embeddings, axis=1))),
            "mean_similarity": float(np.mean(np.dot(embeddings, embeddings.T)))
        }
        return stats


# Improved fallback logic for loading image models
def load_image_model(model_candidates):
    """
    Attempts to load an image model from a list of candidates.
    Returns the first successfully loaded model or None if all fail.
    """
    for model_name in model_candidates:
        try:
            logger.info(f"Attempting to load image model: {model_name}")
            model = SentenceTransformer(model_name)
            logger.info(f"Successfully loaded image model: {model_name}")
            return model
        except Exception as e:
            logger.warning(f"Failed to load image model '{model_name}': {e}")
    logger.error("All image model candidates failed to load.")
    return None

# Example usage and testing
if __name__ == "__main__":
    # Test the embedding generator
    generator = TextEmbeddingGenerator(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=32,
        use_gpu=True,
        normalize=True
    )
    
    # Example: Generate embeddings for captioned frames
    # from frame_extractor import VideoFrameExtractor
    # from caption_generator import BlipCaptionGenerator
    # 
    # # Extract frames
    # extractor = VideoFrameExtractor()
    # frames = extractor.extract_frames("sample_video.mp4")
    # 
    # # Generate captions
    # caption_gen = BlipCaptionGenerator()
    # captioned_frames = caption_gen.generate_captions(frames)
    # 
    # # Generate embeddings
    # embedded_frames = generator.generate_embeddings(captioned_frames)
    # 
    # # Get statistics
    # stats = generator.get_embedding_statistics(embedded_frames)
    # print(f"Embedding statistics: {stats}")
    # 
    # # Test similarity search
    # query = "a person walking"
    # results = generator.find_similar(query, embedded_frames, top_k=5)
    # 
    # for ef, score in results:
    #     print(f"Score: {score:.3f} - Caption: {ef.captioned_frame.caption}")
    #     print(f"  Timestamp: {ef.captioned_frame.frame_data.timestamp:.2f}s")