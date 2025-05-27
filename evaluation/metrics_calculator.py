# evaluation/metrics_calculator.py
import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import random
import logging
from sklearn.metrics.pairwise import cosine_similarity # For cosine distance

from config import settings

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """
    Calculates evaluation metrics for face embeddings.
    """
    def __init__(self, faiss_index: faiss.Index, metadata: List[Dict]):
        """
        Args:
            faiss_index: Loaded FAISS index containing all embeddings.
            metadata: List of metadata dictionaries, one for each embedding,
                      in the same order as in the FAISS index.
                      Each dict: {"identity_id": str, "image_path": str}
        """
        self.index = faiss_index
        self.metadata = metadata
        self.all_embeddings = self._reconstruct_all_embeddings()

    def _reconstruct_all_embeddings(self) -> Optional[np.ndarray]:
        """Reconstructs all embeddings from the FAISS index into a NumPy array."""
        if self.index.ntotal == 0:
            logger.warning("FAISS index is empty, cannot reconstruct embeddings.")
            return None
        try:
            embeddings_np = np.zeros((self.index.ntotal, self.index.d), dtype=np.float32)
            for i in range(self.index.ntotal):
                embeddings_np[i, :] = self.index.reconstruct(i)
            logger.debug(f"Reconstructed {embeddings_np.shape[0]} embeddings from FAISS index.")
            return embeddings_np
        except Exception as e:
            logger.error(f"Failed to reconstruct embeddings from FAISS index: {e}")
            return None

    def _calculate_distance(self, emb1: np.ndarray, emb2: np.ndarray, metric: str) -> float:
        """Calculates distance between two embeddings using the specified metric."""
        emb1 = emb1.reshape(1, -1) # Ensure 2D for scikit-learn functions
        emb2 = emb2.reshape(1, -1)

        if metric == "cosine":
            # Cosine distance = 1 - cosine similarity
            # similarity = cosine_similarity(emb1, emb2)[0, 0]
            # return 1 - similarity
            # Using deepface's preferred way for consistency if available, else direct calc
            # For now, direct calculation:
            dot_product = np.dot(emb1.flatten(), emb2.flatten())
            norm_emb1 = np.linalg.norm(emb1)
            norm_emb2 = np.linalg.norm(emb2)
            if norm_emb1 == 0 or norm_emb2 == 0: return 1.0 # Avoid division by zero
            similarity = dot_product / (norm_emb1 * norm_emb2)
            return 1.0 - similarity # ensure distance is non-negative
        elif metric == "euclidean":
            return np.linalg.norm(emb1 - emb2)
        elif metric == "euclidean_l2":
            # Euclidean L2 is just Euclidean distance.
            # If you meant squared Euclidean, then (np.linalg.norm(emb1 - emb2))**2
            return np.linalg.norm(emb1 - emb2)
        # Add "angular" if needed:
        # elif metric == "angular":
        #     dot_product = np.dot(emb1.flatten(), emb2.flatten())
        #     norm_emb1 = np.linalg.norm(emb1)
        #     norm_emb2 = np.linalg.norm(emb2)
        #     if norm_emb1 == 0 or norm_emb2 == 0: return np.pi # Max angular distance
        #     similarity = np.clip(dot_product / (norm_emb1 * norm_emb2), -1.0, 1.0) # Clip for arccos stability
        #     return np.arccos(similarity) # Radians
        else:
            raise ValueError(f"Unsupported distance metric: {metric}")

    def calculate_pairwise_distances(self,
                                     distance_metric: str,
                                     max_positive_pairs_per_id: int = 50,
                                     max_negative_pairs_total: int = 20000
                                     ) -> Tuple[List[float], List[float]]:
        """
        Calculates distances for positive (same identity) and negative (different identity) pairs.

        Returns:
            Tuple (positive_distances, negative_distances)
        """
        if self.all_embeddings is None or len(self.metadata) != self.all_embeddings.shape[0]:
            logger.error("Embeddings not loaded or metadata mismatch. Cannot calculate distances.")
            return [], []

        positive_distances = []
        negative_distances = []

        # Group indices by identity
        identity_to_indices = defaultdict(list)
        for i, meta in enumerate(self.metadata):
            identity_to_indices[meta['identity_id']].append(i)

        # Generate positive pairs
        logger.info(f"Generating positive pairs for distance metric: {distance_metric}")
        for identity_id, indices in identity_to_indices.items():
            if len(indices) < 2:
                continue # Need at least two images for a positive pair

            current_id_pairs = []
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    idx1, idx2 = indices[i], indices[j]
                    current_id_pairs.append((idx1, idx2))

            # Sample if too many pairs for this identity
            if len(current_id_pairs) > max_positive_pairs_per_id:
                current_id_pairs = random.sample(current_id_pairs, max_positive_pairs_per_id)

            for idx1, idx2 in current_id_pairs:
                dist = self._calculate_distance(self.all_embeddings[idx1], self.all_embeddings[idx2], distance_metric)
                positive_distances.append(dist)
        logger.info(f"Generated {len(positive_distances)} positive pair distances.")


        # Generate negative pairs
        logger.info(f"Generating negative pairs for distance metric: {distance_metric}")
        num_embeddings = self.all_embeddings.shape[0]
        if num_embeddings < 2:
            logger.warning("Not enough unique embeddings to form negative pairs.")
            return positive_distances, []

        # Create a list of all possible (index, identity_id) tuples for efficient negative pair generation
        all_indices_with_ids = [(i, meta['identity_id']) for i, meta in enumerate(self.metadata)]
        attempts = 0
        max_attempts = max_negative_pairs_total * 10 # Safety break

        while len(negative_distances) < max_negative_pairs_total and attempts < max_attempts:
            attempts += 1
            # Pick two random distinct indices
            idx1, idx2 = random.sample(range(num_embeddings), 2)

            id1 = self.metadata[idx1]['identity_id']
            id2 = self.metadata[idx2]['identity_id']

            if id1 != id2: # Ensure they are from different people
                dist = self._calculate_distance(self.all_embeddings[idx1], self.all_embeddings[idx2], distance_metric)
                negative_distances.append(dist)
            if attempts % 10000 == 0: # Log progress for large datasets
                 logger.debug(f"Negative pair generation: {len(negative_distances)} / {max_negative_pairs_total} generated after {attempts} attempts.")


        logger.info(f"Generated {len(negative_distances)} negative pair distances.")
        return positive_distances, negative_distances

    def calculate_accuracy_at_threshold(self,
                                        positive_distances: List[float],
                                        negative_distances: List[float],
                                        threshold: float) -> Dict[str, float]:
        """Calculates TP, FP, TN, FN and accuracy based on a given threshold."""
        tp = sum(1 for d in positive_distances if d < threshold)
        fn = sum(1 for d in positive_distances if d >= threshold)
        tn = sum(1 for d in negative_distances if d >= threshold)
        fp = sum(1 for d in negative_distances if d < threshold)

        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # True Positive Rate (TPR)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0 # False Positive Rate

        return {
            "threshold": threshold,
            "accuracy": accuracy,
            "true_positives": tp,
            "false_negatives": fn,
            "true_negatives": tn,
            "false_positives": fp,
            "precision": precision,
            "recall_tpr": recall,
            "fpr": fpr
        }

    def find_optimal_threshold_and_accuracy(self,
                                            positive_distances: List[float],
                                            negative_distances: List[float],
                                            threshold_percentile: int = settings.THRESHOLD_PERCENTILE
                                            ) -> Optional[Dict[str, float]]:
        """
        Determines a threshold from negative pair distances and calculates accuracy.
        The threshold is set at a percentile of negative distances, aiming to correctly
        classify most negative pairs (i.e., their distance is >= threshold).
        """
        if not negative_distances:
            logger.warning("No negative distances provided, cannot determine threshold or accuracy.")
            return None
        if not positive_distances:
            logger.warning("No positive distances provided, cannot determine accuracy robustly.")
            # Fallback or error, for now returning None
            return None

        # Threshold is the value at which (100-percentile)% of negative pairs are correctly classified as different
        # e.g., if threshold_percentile = 5, threshold is the 5th percentile of negative_distances.
        # This means 5% of different people might be considered "same" (FP) at this threshold,
        # and 95% are correctly considered "different".
        threshold = np.percentile(np.array(negative_distances), threshold_percentile)
        logger.info(f"Determined threshold: {threshold:.4f} (at {threshold_percentile}th percentile of negative distances)")

        return self.calculate_accuracy_at_threshold(positive_distances, negative_distances, threshold)