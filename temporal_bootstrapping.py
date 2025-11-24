"""
Temporal Bootstrapping Module
Improves accuracy using temporal consistency, confidence propagation, and frame context analysis
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from scipy.spatial.distance import cosine
from scipy.ndimage import gaussian_filter1d

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ConfidenceScore:
    """Confidence score with multiple components"""
    caption_confidence: float  # Model-generated confidence for caption
    embedding_confidence: float  # Confidence of embedding quality
    temporal_consistency: float  # How consistent with neighboring frames
    combined_score: float  # Weighted combination of all scores


class TemporalBootstrapper:
    """
    Improves accuracy using temporal consistency and confidence propagation
    """
    
    def __init__(self,
                 temporal_window: int = 5,
                 confidence_threshold: float = 0.5,
                 consistency_weight: float = 0.3,
                 smoothing_sigma: float = 1.5):
        """
        Initialize temporal bootstrapper
        
        Args:
            temporal_window: Number of frames to consider for temporal consistency
            confidence_threshold: Minimum confidence score to keep results
            consistency_weight: Weight for temporal consistency in combined score
            smoothing_sigma: Sigma for Gaussian smoothing of confidence scores
        """
        self.temporal_window = temporal_window
        self.confidence_threshold = confidence_threshold
        self.consistency_weight = consistency_weight
        self.smoothing_sigma = smoothing_sigma
    
    def propagate_confidence_scores(self,
                                   embeddings: np.ndarray,
                                   caption_confidences: List[float],
                                   frame_ids: List[str]) -> List[ConfidenceScore]:
        """
        Propagate and smooth confidence scores across temporal dimension
        
        Args:
            embeddings: Array of embeddings [N, embedding_dim]
            caption_confidences: Per-frame caption generation confidences
            frame_ids: Frame identifiers for temporal ordering
            
        Returns:
            List of ConfidenceScore objects
        """
        n_frames = len(embeddings)
        
        # Compute embedding quality scores (based on embedding magnitude/characteristics)
        embedding_confidences = self._compute_embedding_confidences(embeddings)
        
        # Compute temporal consistency scores
        temporal_scores = self._compute_temporal_consistency(embeddings, frame_ids)
        
        # Smooth confidence scores temporally
        caption_conf_smooth = gaussian_filter1d(
            caption_confidences,
            sigma=self.smoothing_sigma
        )
        embedding_conf_smooth = gaussian_filter1d(
            embedding_confidences,
            sigma=self.smoothing_sigma
        )
        temporal_conf_smooth = gaussian_filter1d(
            temporal_scores,
            sigma=self.smoothing_sigma
        )
        
        # Combine scores with weighting
        confidence_scores = []
        for i in range(n_frames):
            # Normalize weights to sum to 1
            combined = (0.4 * caption_conf_smooth[i] +
                       0.3 * embedding_conf_smooth[i] +
                       self.consistency_weight * temporal_conf_smooth[i])
            
            score = ConfidenceScore(
                caption_confidence=float(caption_conf_smooth[i]),
                embedding_confidence=float(embedding_conf_smooth[i]),
                temporal_consistency=float(temporal_conf_smooth[i]),
                combined_score=float(combined)
            )
            confidence_scores.append(score)
        
        logger.info(f"Propagated confidence scores for {n_frames} frames")
        logger.info(f"Mean confidence: {np.mean([s.combined_score for s in confidence_scores]):.3f}")
        
        return confidence_scores
    
    def _compute_embedding_confidences(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute embedding quality scores based on embedding characteristics
        
        Args:
            embeddings: Array of embeddings [N, embedding_dim]
            
        Returns:
            Array of confidence scores [0, 1]
        """
        # Norm-based confidence: embeddings with good norm are more confident
        norms = np.linalg.norm(embeddings, axis=1)
        norm_conf = (norms - norms.min()) / (norms.max() - norms.min() + 1e-8)
        
        # Entropy-based confidence: less chaotic embeddings are more confident
        # Measure how concentrated the embedding values are
        entropy = -np.sum(embeddings ** 2 * np.log(np.abs(embeddings) + 1e-8), axis=1)
        entropy_norm = 1.0 / (1.0 + entropy)  # Convert to [0, 1]
        
        # Combined embedding confidence
        embedding_conf = 0.6 * norm_conf + 0.4 * entropy_norm
        
        return np.clip(embedding_conf, 0, 1)
    
    def _compute_temporal_consistency(self,
                                     embeddings: np.ndarray,
                                     frame_ids: List[str]) -> np.ndarray:
        """
        Compute temporal consistency scores based on neighboring frames
        
        Args:
            embeddings: Array of embeddings [N, embedding_dim]
            frame_ids: Frame identifiers
            
        Returns:
            Array of temporal consistency scores [0, 1]
        """
        n_frames = len(embeddings)
        temporal_scores = np.zeros(n_frames)
        
        for i in range(n_frames):
            # Define temporal window
            start_idx = max(0, i - self.temporal_window)
            end_idx = min(n_frames, i + self.temporal_window + 1)
            
            # Compute average similarity to neighboring embeddings
            similarities = []
            for j in range(start_idx, end_idx):
                if i != j:
                    sim = 1.0 - cosine(embeddings[i], embeddings[j])
                    similarities.append(sim)
            
            # Temporal consistency: average similarity to neighbors
            if similarities:
                temporal_scores[i] = np.mean(similarities)
            else:
                temporal_scores[i] = 0.5  # Default for isolated frames
        
        # Normalize to [0, 1]
        temporal_scores = np.clip(temporal_scores, 0, 1)
        
        return temporal_scores
    
    def filter_by_confidence(self,
                           results: List[Dict],
                           confidence_scores: Dict[str, ConfidenceScore]) -> List[Dict]:
        """
        Filter search results based on confidence scores
        
        Args:
            results: Search results from Pinecone
            confidence_scores: Mapping of frame_id to ConfidenceScore
            
        Returns:
            Filtered results with confidence scores attached
        """
        filtered_results = []
        
        for result in results:
            frame_id = result.get('frame_id')
            
            if frame_id in confidence_scores:
                conf_score = confidence_scores[frame_id]
                
                # Only include if confidence exceeds threshold
                if conf_score.combined_score >= self.confidence_threshold:
                    result['confidence'] = conf_score
                    result['combined_confidence'] = conf_score.combined_score
                    filtered_results.append(result)
            else:
                # Include results without confidence info (backward compatibility)
                filtered_results.append(result)
        
        logger.info(f"Filtered {len(results)} results to {len(filtered_results)} "
                   f"by confidence threshold {self.confidence_threshold}")
        
        return filtered_results
    
    def boost_high_confidence_results(self,
                                     results: List[Dict],
                                     boost_factor: float = 1.5) -> List[Dict]:
        """
        Boost ranking of high-confidence results
        
        Args:
            results: Search results with similarity scores
            boost_factor: Factor to multiply scores for high-confidence results
            
        Returns:
            Re-ranked results
        """
        boosted_results = []
        
        for result in results:
            if 'combined_confidence' in result:
                conf = result['combined_confidence']
                # Boost if high confidence
                if conf > 0.7:
                    result['boosted_score'] = result.get('similarity_score', 0) * boost_factor
                else:
                    result['boosted_score'] = result.get('similarity_score', 0)
            
            boosted_results.append(result)
        
        # Re-sort by boosted score
        boosted_results.sort(key=lambda x: x.get('boosted_score', 0), reverse=True)
        
        return boosted_results
    
    def intra_video_temporal_smoothing(self,
                                      video_frames: List[Dict],
                                      search_query_embedding: np.ndarray,
                                      caption_embeddings: Dict[str, np.ndarray],
                                      image_embeddings: Optional[Dict[str, np.ndarray]] = None) -> List[Dict]:
        """
        Apply temporal smoothing within a single video for consistent results
        
        Args:
            video_frames: Frames from a single video in temporal order
            search_query_embedding: Query embedding vector
            caption_embeddings: Dict mapping frame_id to caption embedding
            image_embeddings: Optional dict mapping frame_id to image embedding
            
        Returns:
            Temporally smoothed and ranked results
        """
        if not video_frames:
            return []
        
        results = []
        
        for i, frame in enumerate(video_frames):
            frame_id = frame.get('frame_id')
            
            # Compute similarity to query
            caption_sim = 0
            image_sim = 0
            
            if frame_id in caption_embeddings:
                caption_emb = caption_embeddings[frame_id]
                caption_sim = 1.0 - cosine(search_query_embedding, caption_emb)
            
            if image_embeddings and frame_id in image_embeddings:
                image_emb = image_embeddings[frame_id]
                image_sim = 1.0 - cosine(search_query_embedding, image_emb)
            
            # Combine caption and image similarity
            combined_sim = 0.6 * caption_sim + 0.4 * image_sim if image_embeddings else caption_sim
            
            # Apply temporal context boost from neighbors
            neighbor_boost = 0
            for j in range(max(0, i - 2), min(len(video_frames), i + 3)):
                if j != i:
                    neighbor_frame_id = video_frames[j].get('frame_id')
                    if neighbor_frame_id in caption_embeddings:
                        neighbor_emb = caption_embeddings[neighbor_frame_id]
                        neighbor_sim = 1.0 - cosine(search_query_embedding, neighbor_emb)
                        neighbor_boost += neighbor_sim * 0.1  # Small contribution
            
            final_score = combined_sim + neighbor_boost
            
            result = frame.copy()
            result['combined_score'] = final_score
            result['caption_similarity'] = caption_sim
            result['image_similarity'] = image_sim
            results.append(result)
        
        # Sort by final score
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return results
    
    def get_confidence_report(self,
                             confidence_scores: List[ConfidenceScore]) -> Dict:
        """
        Generate a report of confidence score statistics
        
        Args:
            confidence_scores: List of ConfidenceScore objects
            
        Returns:
            Dictionary with statistics
        """
        if not confidence_scores:
            return {}
        
        combined_scores = [s.combined_score for s in confidence_scores]
        caption_scores = [s.caption_confidence for s in confidence_scores]
        embedding_scores = [s.embedding_confidence for s in confidence_scores]
        temporal_scores = [s.temporal_consistency for s in confidence_scores]
        
        report = {
            'total_frames': len(confidence_scores),
            'combined_score': {
                'mean': float(np.mean(combined_scores)),
                'std': float(np.std(combined_scores)),
                'min': float(np.min(combined_scores)),
                'max': float(np.max(combined_scores)),
                'above_threshold': sum(1 for s in combined_scores if s >= self.confidence_threshold)
            },
            'caption_confidence': {
                'mean': float(np.mean(caption_scores)),
                'std': float(np.std(caption_scores)),
            },
            'embedding_confidence': {
                'mean': float(np.mean(embedding_scores)),
                'std': float(np.std(embedding_scores)),
            },
            'temporal_consistency': {
                'mean': float(np.mean(temporal_scores)),
                'std': float(np.std(temporal_scores)),
            }
        }
        
        return report
