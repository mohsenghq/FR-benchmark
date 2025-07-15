# processing/embedding_processor.py
import faiss
import numpy as np
import os
import pickle
import logging
import time
from typing import List, Dict, Tuple, Optional

from config import settings
from deepface_integration.deepface_handler import DeepFaceHandler, DeepFaceHandlerError
from common import utils

logger = logging.getLogger(__name__)

class EmbeddingProcessor:
    """
    Processes a dataset of pre-cropped face images to generate embeddings
    and stores them in a FAISS index.
    """
    def __init__(self,
                 deepface_handler_instance: DeepFaceHandler,
                 faiss_dir: str = settings.FAISS_INDEXES_DIR):
        self.handler = deepface_handler_instance
        self.faiss_dir = faiss_dir
        utils.ensure_dir_exists(self.faiss_dir)

    def _get_faiss_paths(self, model_name: str) -> Tuple[str, str]: # Detector no longer part of filename base for FAISS
        """Generates file paths for FAISS index and its metadata."""
        # Detector is fixed during cropping, so not needed in FAISS filename for embeddings
        filename_base = f"{model_name}".replace('-', '_').lower()
        index_path = os.path.join(self.faiss_dir, f"{filename_base}.index")
        metadata_path = os.path.join(self.faiss_dir, f"{filename_base}_metadata.pkl")
        return index_path, metadata_path

    def generate_and_store_embeddings(self,
                                      cropped_identities_map: Dict[str, List[str]], # Expects paths to cropped images
                                      model_name: str,
                                      force_regenerate: bool = False
                                      ) -> Optional[Tuple[str, str, int, float]]:
        """
        Generates embeddings for all pre-cropped images using the specified model
        and stores them in a FAISS index.

        Args:
            cropped_identities_map: Dict mapping identity_id to list of *cropped* image paths.
            model_name: The facial recognition model to use.
            force_regenerate: If True, regenerate embeddings even if an index file exists.

        Returns:
            A tuple (faiss_index_path, metadata_path, num_embeddings, avg_embedding_time_ms)
            or None if generation fails.
        """
        index_path, metadata_path = self._get_faiss_paths(model_name) # Detector backend removed from here

        # --- Check for dataset changes ---
        dataset_image_paths = set()
        for paths in cropped_identities_map.values():
            dataset_image_paths.update(paths)
        dataset_image_paths = set(sorted(dataset_image_paths))

        need_regenerate = force_regenerate
        loaded_index = None
        if not force_regenerate and os.path.exists(index_path) and os.path.exists(metadata_path):
            logger.info(f"FAISS index and metadata already exist for {model_name}. Checking for dataset changes.")
            try:
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                stored_image_paths = set(sorted([m['image_path'] for m in metadata]))
                if stored_image_paths != dataset_image_paths:
                    logger.info("Dataset has changed (images added/removed). Regenerating FAISS index and metadata.")
                    need_regenerate = True
                else:
                    loaded_index = faiss.read_index(index_path)
                    if loaded_index.ntotal == len(metadata):
                        logger.info(f"Loaded {loaded_index.ntotal} embeddings from existing files.")
                        return index_path, metadata_path, loaded_index.ntotal, 0.0 # Placeholder for avg time
                    else:
                        logger.warning("Mismatch between loaded index and metadata. Regenerating.")
                        need_regenerate = True
            except Exception as e:
                logger.warning(f"Could not load existing FAISS data for {model_name}: {e}. Regenerating.")
                need_regenerate = True

        if not need_regenerate:
            if os.path.exists(index_path) and os.path.exists(metadata_path):
                logger.info(f"No need to regenerate FAISS index for {model_name}.")
                return index_path, metadata_path, 0, 0.0
            else:
                logger.warning(f"FAISS index or metadata missing for {model_name}, forcing regeneration.")
                need_regenerate = True

        logger.info(f"Generating embeddings for model: {model_name} using pre-cropped faces.")
        embeddings_list = []
        metadata_list = []
        embedding_times = []
        total_cropped_images = sum(len(paths) for paths in cropped_identities_map.values())
        processed_images_count = 0

        if total_cropped_images == 0:
            logger.warning(f"No pre-cropped images provided for model {model_name}. Cannot generate embeddings.")
            return None

        for identity_id, cropped_image_paths in cropped_identities_map.items():
            for cropped_image_path in cropped_image_paths:
                processed_images_count += 1
                logger.debug(f"Processing cropped image {processed_images_count}/{total_cropped_images}: {cropped_image_path} for identity {identity_id}")
                try:
                    start_time = time.monotonic()
                    # Pass the path to the pre-cropped image
                    embedding = self.handler.get_embedding(
                        cropped_image_path_or_array=cropped_image_path,
                        model_name=model_name
                    )
                    elapsed_time_ms = (time.monotonic() - start_time) * 1000
                    embedding_times.append(elapsed_time_ms)

                    if embedding:
                        embeddings_list.append(embedding)
                        metadata_list.append({
                            "identity_id": identity_id,
                            "image_path": cropped_image_path # Store path to cropped image
                        })
                        logger.debug(f"Generated embedding for {cropped_image_path} in {elapsed_time_ms:.2f} ms")
                    else:
                        logger.warning(f"No embedding returned for {cropped_image_path} (identity: {identity_id}). Skipping.")

                except DeepFaceHandlerError as e:
                    logger.error(f"DeepFace Handler error processing {cropped_image_path} for {identity_id}: {e}. Skipping.")
                except Exception as e:
                    logger.error(f"Unexpected error processing {cropped_image_path} for {identity_id}: {e}. Skipping.")

        if not embeddings_list:
            logger.error(f"No embeddings were successfully generated for model {model_name}.")
            return None

        embeddings_np = np.array(embeddings_list, dtype=np.float32)
        if embeddings_np.ndim != 2 or embeddings_np.shape[0] == 0:
            logger.error(f"Embeddings array is not valid for FAISS for model {model_name}. Shape: {embeddings_np.shape}")
            return None

        dimension = embeddings_np.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_np)

        logger.info(f"Saving FAISS index ({index.ntotal} vectors, dim: {dimension}) to {index_path}")
        faiss.write_index(index, index_path)

        logger.info(f"Saving metadata ({len(metadata_list)} entries) to {metadata_path}")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata_list, f)

        avg_embedding_time = np.mean(embedding_times) if embedding_times else 0
        logger.info(f"Finished generating {index.ntotal} embeddings for {model_name}. Avg time/image: {avg_embedding_time:.2f} ms.")
        return index_path, metadata_path, index.ntotal, avg_embedding_time

    def load_embeddings_and_metadata(self, model_name: str) -> Optional[Tuple[faiss.Index, List[Dict]]]:
        """Loads a FAISS index and its corresponding metadata."""
        index_path, metadata_path = self._get_faiss_paths(model_name) # Detector removed
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            logger.error(f"FAISS index or metadata not found for {model_name} at {index_path}/{metadata_path}")
            return None
        try:
            index = faiss.read_index(index_path)
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            if index.ntotal != len(metadata):
                logger.error(f"Mismatch between FAISS index size ({index.ntotal}) and metadata length ({len(metadata)}).")
                return None
            logger.info(f"Loaded FAISS index ({index.ntotal} vectors) and metadata for {model_name}.")
            return index, metadata
        except Exception as e:
            logger.error(f"Error loading FAISS data for {model_name}: {e}")
            return None
