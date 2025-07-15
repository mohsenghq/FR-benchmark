# main.py
import logging
import time
import numpy as np

from deepface import DeepFace # Import DeepFace for pre-building models

from config import settings
from dataset.loader import DatasetLoader
from deepface_integration.deepface_handler import DeepFaceHandler
from processing.embedding_processor import EmbeddingProcessor
from evaluation.metrics_calculator import MetricsCalculator
from visualizing.results_visualizer import ResultsVisualizer
from common import utils
from preprocessing.face_cropper import FaceCropper

# Initialize once
visualizer = ResultsVisualizer(plots_dir=settings.PLOTS_DIR, reports_dir=settings.REPORTS_DIR)
visualizer.setup_logging(log_file=settings.LOG_FILENAME)

logger = logging.getLogger(__name__)

def preload_models(models_to_test: list):
    """Attempts to pre-load/build all specified models to catch issues early."""
    logger.info("Attempting to pre-load/build specified models...")
    successfully_loaded_models = []
    for model_name in models_to_test:
        try:
            logger.info(f"Pre-building model: {model_name}")
            DeepFace.build_model(model_name)
            logger.info(f"Successfully pre-built model: {model_name}")
            successfully_loaded_models.append(model_name)
        except Exception as e:
            logger.error(f"Failed to pre-build model {model_name}: {e}. This model will be skipped.")
    
    if not successfully_loaded_models:
        logger.error("No models could be successfully pre-loaded. Exiting pipeline.")
        return None # Indicate failure
    
    # Return only the models that were successfully loaded
    return successfully_loaded_models


def run_evaluation_pipeline():
    """
    Orchestrates the entire face recognition model evaluation pipeline,
    including a pre-processing step for face cropping and model pre-loading.
    """
    use_super_resolution = settings.USE_SUPER_RESOLUTION  # Change to False to disable SR
    logger.info("Starting Face Recognition Model Evaluation Pipeline...")
    # utils.ensure_dir_exists(settings.OUTPUT_BASE_DIR)
    utils.ensure_dir_exists(settings.CROPPED_FACES_DIR)
    utils.ensure_dir_exists(settings.FAISS_INDEXES_DIR)
    utils.ensure_dir_exists(settings.RUNS_BASE_DIR)
    utils.ensure_dir_exists(settings.RUN_FOLDER)
    utils.ensure_dir_exists(settings.LOGS_DIR)
    utils.ensure_dir_exists(settings.PLOTS_DIR)
    utils.ensure_dir_exists(settings.REPORTS_DIR)

    # --- 0. Pre-load Models ---
    # This will also filter MODELS_TO_TEST to only include successfully loaded ones
    active_models_to_test = preload_models(settings.MODELS_TO_TEST)
    if not active_models_to_test:
        logger.critical("Model pre-loading failed for all specified models. Cannot continue.")
        return

    # --- 1. Load Raw Dataset Paths ---
    logger.info(f"Loading raw dataset paths from: {settings.DATASET_ROOT_DIR}")
    dataset_loader = DatasetLoader(dataset_root=settings.DATASET_ROOT_DIR)
    try:
        raw_identities_map = dataset_loader.get_identities_with_images()
    except FileNotFoundError:
        logger.warning(f"Raw dataset folder {settings.DATASET_ROOT_DIR} not found. Will try to use cropped faces only.")
        raw_identities_map = None

    # --- 1b. Fallback: Use Cropped Faces Directly if Raw Images Not Available ---
    if not raw_identities_map:
        logger.warning("No identities found in the raw dataset. Attempting to use cropped faces folder directly.")
        cropped_faces_loader = DatasetLoader(dataset_root=settings.CROPPED_FACES_DIR)
        try:
            cropped_identities_map = cropped_faces_loader.get_identities_with_images()
        except FileNotFoundError:
            logger.error(f"Cropped faces folder {settings.CROPPED_FACES_DIR} not found. Exiting pipeline.")
            return
        if not cropped_identities_map:
            logger.error("No identities found in the cropped faces folder. Exiting pipeline.")
            return
        logger.info(f"Loaded {sum(len(paths) for paths in cropped_identities_map.values())} cropped faces for embedding.")
    else:
        # --- 2. Pre-processing: Detect and Crop Faces ---
        logger.info(f"Starting face cropping using detector: {settings.SELECTED_DETECTOR_BACKEND}")
        logger.info(f"Raw images will be processed and cropped faces saved to: {settings.CROPPED_FACES_DIR}")
        face_cropper = FaceCropper(
            raw_dataset_root=settings.DATASET_ROOT_DIR,
            cropped_dataset_root=settings.CROPPED_FACES_DIR,
            detector_backend=settings.SELECTED_DETECTOR_BACKEND,
            use_super_resolution=use_super_resolution
        )
        cropped_identities_map = face_cropper.process_and_save_cropped_faces(raw_identities_map)

        if not cropped_identities_map:
            logger.error("No faces were successfully cropped from the dataset. Cannot proceed. Exiting.")
            return
        logger.info(f"Successfully cropped faces. {sum(len(paths) for paths in cropped_identities_map.values())} faces available for embedding.")

    # --- 3. Initialize DeepFace Handler ---
    logger.info("Initializing DeepFace Handler for direct library usage on pre-cropped faces.")
    deepface_handler = DeepFaceHandler()

    # --- 4. Initialize Embedding Processor ---
    embedding_processor = EmbeddingProcessor(deepface_handler_instance=deepface_handler)

    all_experiment_results = []

    # --- 5. Loop Through Successfully Pre-loaded Models ---
    for model_name in active_models_to_test: # Use the filtered list
        logger.info(f"===== Processing Model: {model_name} using pre-cropped faces =====")
        pipeline_start_time_model = time.monotonic()

        embedding_data = embedding_processor.generate_and_store_embeddings(
            cropped_identities_map=cropped_identities_map,
            model_name=model_name,
            force_regenerate=False
        )

        if not embedding_data:
            logger.error(f"Failed to generate/load embeddings for model {model_name}. Skipping this model.")
            all_experiment_results.append({
                "model": model_name,
                "cropping_detector": settings.SELECTED_DETECTOR_BACKEND, # Consistent key
                "distance_metric": "N/A", "num_embeddings": 0,
                "accuracy": "Error", "threshold": "Error", "precision": "Error",
                "recall_tpr": "Error", "fpr": "Error",
                "avg_embedding_time_ms": "Error", "evaluation_time_s": "Error"
            })
            continue

        faiss_index_path, metadata_path, num_embeddings, avg_embedding_time_ms = embedding_data
        logger.info(f"Embeddings for {model_name} ready. Index: {faiss_index_path}, Metadata: {metadata_path}")
        logger.info(f"Total embeddings: {num_embeddings}, Avg. embedding time: {avg_embedding_time_ms:.2f} ms/image")

        loaded_data = embedding_processor.load_embeddings_and_metadata(model_name)
        if not loaded_data:
            logger.error(f"Could not load FAISS index/metadata for {model_name} for evaluation. Skipping metrics.")
            continue
        faiss_index_obj, metadata_list = loaded_data

        metrics_calc = MetricsCalculator(faiss_index=faiss_index_obj, metadata=metadata_list)

        for dist_metric_name in settings.DISTANCE_METRICS_TO_TEST:
            logger.info(f"--- Evaluating Model: {model_name} with Distance Metric: {dist_metric_name} ---")
            eval_start_time = time.monotonic()

            positive_distances, negative_distances = metrics_calc.calculate_pairwise_distances(
                distance_metric=dist_metric_name
            )

            accuracy_info = {}
            if not positive_distances and not negative_distances:
                 logger.warning(f"No distances generated for {model_name} - {dist_metric_name}. Skipping accuracy calculation.")
                 accuracy_info = {"accuracy": "N/A", "threshold": "N/A", "precision": "N/A", "recall_tpr": "N/A", "fpr": "N/A"}
            elif not negative_distances:
                 logger.warning(f"No negative distances for {model_name} - {dist_metric_name}. Cannot determine threshold automatically.")
                 accuracy_info = {"accuracy": "N/A", "threshold": "N/A (No neg pairs)", "precision": "N/A", "recall_tpr": "N/A", "fpr": "N/A"}
            else:
                accuracy_info = metrics_calc.find_optimal_threshold_and_accuracy(
                    positive_distances=positive_distances,
                    negative_distances=negative_distances,
                    threshold_percentile=settings.THRESHOLD_PERCENTILE
                )
                if not accuracy_info: # Should only happen if positive_distances was empty and negative_distances was not
                    accuracy_info = {"accuracy": "N/A", "threshold": "N/A", "precision": "N/A", "recall_tpr": "N/A", "fpr": "N/A"}


            eval_duration_s = time.monotonic() - eval_start_time
            logger.info(f"Evaluation for {model_name} ({dist_metric_name}) completed in {eval_duration_s:.2f}s.")

            visualizer.log_experiment_summary(
                model_name=model_name,
                detector=settings.SELECTED_DETECTOR_BACKEND, # This is the cropping detector
                metric=dist_metric_name,
                accuracy_results=accuracy_info,
                avg_embedding_time_ms=avg_embedding_time_ms,
                evaluation_duration_s=eval_duration_s,
                num_embeddings=num_embeddings
            )

            if accuracy_info and 'threshold' in accuracy_info and isinstance(accuracy_info['threshold'], (int, float, np.float32, np.float64)):
                visualizer.plot_distance_distributions(
                    positive_distances, negative_distances,
                    model_name, dist_metric_name,
                    threshold=accuracy_info['threshold'],
                    filename_suffix=f"cropped_with_{settings.SELECTED_DETECTOR_BACKEND}"
                )
            else:
                 logger.warning(f"Skipping plotting for {model_name} - {dist_metric_name} due to missing threshold or distances.")

            all_experiment_results.append({
                "model": model_name,
                "cropping_detector": settings.SELECTED_DETECTOR_BACKEND, # Consistent key
                "distance_metric": dist_metric_name,
                "num_embeddings": num_embeddings,
                "avg_embedding_time_ms": round(avg_embedding_time_ms, 2) if isinstance(avg_embedding_time_ms, float) else avg_embedding_time_ms,
                "evaluation_time_s": round(eval_duration_s, 2),
                "threshold": accuracy_info.get('threshold', "N/A"),
                "accuracy": accuracy_info.get('accuracy', "N/A"),
                "true_positives": accuracy_info.get('true_positives', "N/A"),
                "false_negatives": accuracy_info.get('false_negatives', "N/A"),
                "true_negatives": accuracy_info.get('true_negatives', "N/A"),
                "false_positives": accuracy_info.get('false_positives', "N/A"),
                "precision": accuracy_info.get('precision', "N/A"),
                "recall_tpr": accuracy_info.get('recall_tpr', "N/A"),
                "fpr": accuracy_info.get('fpr', "N/A"),
            })
        model_processing_duration = time.monotonic() - pipeline_start_time_model
        logger.info(f"Total processing time for model {model_name} (embedding + eval): {model_processing_duration:.2f}s")

    logger.info("All models and metrics processed. Generating final summary report...")
    if not all_experiment_results:
        logger.warning("No experiment results were generated. Skipping final report table.")
    else:
        visualizer.generate_summary_report_table(all_experiment_results)

    logger.info("Face Recognition Model Evaluation Pipeline Finished.")

if __name__ == "__main__":
    try:
        run_evaluation_pipeline()
    except Exception as e:
        logger.critical(f"Critical error in pipeline execution: {e}", exc_info=True)

