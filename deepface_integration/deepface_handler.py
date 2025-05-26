# deepface_integration/deepface_handler.py
import logging
from typing import Optional, List
from deepface import DeepFace
import numpy as np # Added for type checking if needed

logger = logging.getLogger(__name__)

class DeepFaceHandlerError(Exception):
    """Custom exception for errors during DeepFace library calls."""
    pass

class DeepFaceHandler:
    """
    A handler to interact with the DeepFace library directly for face embedding.
    Assumes input images are already cropped and aligned faces.
    """
    def __init__(self):
        logger.info("DeepFaceHandler initialized for direct library calls on pre-cropped faces.")

    def get_embedding(self,
                      cropped_image_path_or_array: str, # Can be path to a pre-cropped image or a numpy array
                      model_name: str) -> Optional[List[float]]:
        """
        Uses the DeepFace library to get an embedding from a pre-cropped face image.

        Args:
            cropped_image_path_or_array: Path to the pre-cropped and aligned face image
                                         (or a NumPy array of the face).
            model_name: Name of the facial recognition model to use.

        Returns:
            A list of floats representing the embedding, or None if an error occurs.
        """
        logger.debug(f"Requesting embedding for pre-cropped face {type(cropped_image_path_or_array)} with model {model_name}.")
        try:
            # For pre-cropped faces, detection and alignment should be skipped.
            # DeepFace.represent handles numpy arrays directly by skipping detection/alignment.
            # If it's a path to an already cropped image, we explicitly tell it not to re-detect/align.
            embedding_objs = DeepFace.represent(
                img_path=cropped_image_path_or_array,
                model_name=model_name,
                detector_backend='skip', # Explicitly skip detection; 'skip' is a valid backend
                enforce_detection=False, # Not needed as we provide a face
                align=False              # Already aligned during cropping
            )

            if embedding_objs and isinstance(embedding_objs, list) and len(embedding_objs) > 0:
                embedding_data = embedding_objs[0]
                if "embedding" in embedding_data:
                    embedding = embedding_data["embedding"]
                    logger.debug(f"Successfully retrieved embedding. Length: {len(embedding)}")
                    return embedding
                else:
                    logger.warning(f"No 'embedding' key in DeepFace response object. Object: {embedding_data}")
                    raise DeepFaceHandlerError("Embedding key not found in DeepFace response.")
            else:
                logger.warning(f"Embedding failed. DeepFace.represent returned: {embedding_objs}")
                raise DeepFaceHandlerError("Embedding failed; DeepFace.represent returned no valid objects.")

        except ValueError as ve:
            logger.warning(f"DeepFace ValueError for {type(cropped_image_path_or_array)} (model: {model_name}): {str(ve)}")
            raise DeepFaceHandlerError(f"DeepFace library ValueError: {str(ve)}")
        except Exception as e:
            logger.error(f"An unexpected error occurred with DeepFace library for {type(cropped_image_path_or_array)} (model: {model_name}): {e}", exc_info=True)
            raise DeepFaceHandlerError(f"Unexpected DeepFace library error: {e}")
