# preprocessing/face_cropper.py
import os
import cv2 # OpenCV for saving images
import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
from deepface import DeepFace
import glob # For checking existing files
from preprocessing.srcnn_superres import enhance_with_srcnn

from config import settings
from common import utils

logger = logging.getLogger(__name__)

class FaceCropper:
    """
    Detects, crops, and aligns faces from a dataset of raw images.
    Saves the cropped faces to a new directory structure inside the dataset root, preserving identity labels.
    """
    def __init__(self,
                 raw_dataset_root: str,
                 cropped_dataset_root: str,
                 detector_backend: str = settings.SELECTED_DETECTOR_BACKEND,
                 skip_existing: bool = True,
                 use_super_resolution: bool = True): # Added use_super_resolution option
        """
        Args:
            raw_dataset_root: Path to the root directory of the original dataset.
            cropped_dataset_root: Path to the directory where cropped faces will be saved.
            detector_backend: The face detector to use (e.g., 'mtcnn', 'opencv').
            skip_existing: If True, skips processing raw images if their cropped versions already exist.
            use_super_resolution: If True, applies SRCNN super-resolution to cropped faces before saving.
        """
        self.raw_dataset_root = raw_dataset_root
        self.cropped_dataset_root = cropped_dataset_root
        self.detector_backend = detector_backend
        self.skip_existing = skip_existing # Store the option
        self.use_super_resolution = use_super_resolution # Store the option
        utils.ensure_dir_exists(self.cropped_dataset_root)
        logger.info(f"FaceCropper initialized. Raw data: '{raw_dataset_root}', Cropped data output: '{cropped_dataset_root}', Detector: '{detector_backend}', Skip existing: {self.skip_existing}, Use SR: {self.use_super_resolution}")

    def process_and_save_cropped_faces(self, identities_map: Dict[str, List[str]], use_super_resolution: bool = None) -> Dict[str, List[str]]:
        """
        Processes all raw images, extracts faces, and saves them.
        Optionally skips processing if cropped images already exist.
        Optionally applies super-resolution enhancement.

        Args:
            identities_map: A dictionary mapping identity_id to a list of raw image paths.
                            Example: {"person_A": ["path/to/img1.jpg", ...]}
            use_super_resolution: If set, overrides the instance's use_super_resolution setting for this call.

        Returns:
            A dictionary mapping identity_id to a list of paths to *cropped* image files.
            Example: {"person_A": ["path/to/cropped/img1_face0.jpg", ...]}
            Returns an empty dict if no faces are successfully cropped.
        """
        if use_super_resolution is None:
            use_super_resolution = self.use_super_resolution
        cropped_identities_map: Dict[str, List[str]] = {}
        total_raw_images = sum(len(paths) for paths in identities_map.values())
        processed_images_count = 0
        successfully_cropped_count = 0
        skipped_due_to_existence_count = 0
        failed_to_crop_count = 0

        logger.info(f"Starting face cropping process for {total_raw_images} raw images. Skip existing set to: {self.skip_existing}")

        for identity_id, raw_image_paths in identities_map.items():
            identity_output_dir = os.path.join(self.cropped_dataset_root, identity_id)
            utils.ensure_dir_exists(identity_output_dir)
            
            if identity_id not in cropped_identities_map:
                cropped_identities_map[identity_id] = []


            for raw_image_path in raw_image_paths:
                processed_images_count += 1
                original_filename_no_ext = os.path.splitext(os.path.basename(raw_image_path))[0]
                
                potential_first_cropped_face_path = os.path.join(identity_output_dir, f"{original_filename_no_ext}_face0.png")

                if self.skip_existing and os.path.exists(potential_first_cropped_face_path):
                    existing_cropped_files = glob.glob(os.path.join(identity_output_dir, f"{original_filename_no_ext}_face*.png"))
                    if existing_cropped_files:
                        # logger.info(f"Skipping raw image {raw_image_path} as cropped version(s) like {potential_first_cropped_face_path} already exist. Found {len(existing_cropped_files)} existing crops.")
                        cropped_identities_map[identity_id].extend(existing_cropped_files)
                        successfully_cropped_count += len(existing_cropped_files) 
                        skipped_due_to_existence_count += 1
                        continue 
                    else:
                        pass
                        # logger.info(f"Skipping raw image {raw_image_path} (checked {potential_first_cropped_face_path}), but no matching files found with glob. Will attempt processing.")


                logger.debug(f"Processing raw image ({processed_images_count}/{total_raw_images}): {raw_image_path}")
                try:
                    extracted_data = DeepFace.extract_faces(
                        img_path=raw_image_path,
                        detector_backend=self.detector_backend,
                        align=True,
                        enforce_detection=True,
                    )

                    if not extracted_data:
                        logger.warning(f"No faces extracted from {raw_image_path} for identity {identity_id} (enforce_detection was True). Skipping.")
                        failed_to_crop_count +=1
                        continue

                    for i, face_data in enumerate(extracted_data):
                        face_np_bgr = face_data['face']
                        
                        if face_np_bgr.dtype == np.float32 or face_np_bgr.dtype == np.float64:
                             face_np_bgr = (face_np_bgr * 255).astype(np.uint8)

                        # Enhance with SRCNN before saving (if enabled)
                        if use_super_resolution:
                            try:
                                face_np_bgr = enhance_with_srcnn(face_np_bgr)
                            except Exception as e:
                                logger.warning(f"SRCNN enhancement failed for {raw_image_path}: {e}. Saving original crop.")
                                raise e

                        cropped_filename = f"{original_filename_no_ext}_face{i}.png"
                        cropped_image_path = os.path.join(identity_output_dir, cropped_filename)

                        cv2.imwrite(cropped_image_path, face_np_bgr)
                        cropped_identities_map[identity_id].append(cropped_image_path)
                        successfully_cropped_count += 1
                        logger.debug(f"Successfully cropped and saved face to {cropped_image_path}")

                except ValueError as ve:
                    logger.warning(f"ValueError (likely no face detected) for {raw_image_path} (identity {identity_id}): {ve}. Skipping.")
                    failed_to_crop_count +=1
                except Exception as e:
                    logger.error(f"Unexpected error cropping face from {raw_image_path} (identity {identity_id}): {e}", exc_info=True)
                    failed_to_crop_count +=1
            
            if not cropped_identities_map.get(identity_id): 
                if identity_id in cropped_identities_map: 
                     del cropped_identities_map[identity_id]
                logger.warning(f"No faces successfully cropped or found existing for identity: {identity_id}")


        logger.info(f"Face cropping finished. Successfully cropped/found existing: {successfully_cropped_count} faces. Skipped due to prior existence: {skipped_due_to_existence_count} raw images. Failed attempts on new images: {failed_to_crop_count} images.")
        if not cropped_identities_map:
            logger.warning("No faces were successfully cropped or found existing from the entire dataset.")
        return cropped_identities_map