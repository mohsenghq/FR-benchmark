# dataset/loader.py
import os
from collections import defaultdict
import logging

from config import settings # Use the centralized settings

logger = logging.getLogger(__name__)

class DatasetLoader:
    """
    Loads image paths from a dataset directory structured with subfolders
    representing unique identities.
    """
    def __init__(self, dataset_root: str):
        if not os.path.isdir(dataset_root):
            logger.error(f"Dataset root directory not found: {dataset_root}")
            raise FileNotFoundError(f"Dataset root directory not found: {dataset_root}")
        self.dataset_root = dataset_root
        self.supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    def get_identities_with_images(self) -> dict[str, list[str]]:
        """
        Scans the dataset directory and returns a dictionary mapping
        identity names (folder names) to a list of their image file paths.

        Returns:
            dict: {identity_id: [image_path1, image_path2, ...]}
        """
        identities_map = defaultdict(list)
        logger.info(f"Scanning dataset directory: {self.dataset_root}")

        IGNORED_FOLDERS = {"cropped_faces", "faiss_indexes", "results"}

        for identity_name in os.listdir(self.dataset_root):
            if identity_name in IGNORED_FOLDERS:
                logger.info(f"Skipping non-identity folder: {identity_name}")
                continue
            identity_path = os.path.join(self.dataset_root, identity_name)
            if os.path.isdir(identity_path):
                image_count = 0
                for image_file in os.listdir(identity_path):
                    if image_file.lower().endswith(self.supported_extensions):
                        full_image_path = os.path.join(identity_path, image_file)
                        identities_map[identity_name].append(full_image_path)
                        image_count += 1
                if image_count > 0:
                    logger.debug(f"Found {image_count} images for identity: {identity_name}")
                else:
                    logger.warning(f"No supported images found for identity: {identity_name}")
            else:
                logger.warning(f"Skipping non-directory item in dataset root: {identity_name}")

        if not identities_map:
            logger.warning("No identities or images found in the dataset.")
        else:
            logger.info(f"Successfully loaded {len(identities_map)} identities.")
        return dict(identities_map)