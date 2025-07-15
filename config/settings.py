import os
import datetime
import uuid

MODELS_TO_TEST = [
    "VGG-Face", 
    "Facenet", 
    "Facenet512", 
    "OpenFace", 
    "DeepID",
    "ArcFace", 
    "Dlib", 
    "SFace", 
    "GhostFaceNet", 
    # "Buffalo_L", 
    # "DeepFace"
]

# Face Detector (Manual Selection for the initial cropping phase)
SELECTED_DETECTOR_BACKEND = 'mtcnn' # This detector is used ONCE to crop all faces
# 'opencv', 'ssd', 'dlib', 'mtcnn', 'fastmtcnn', 'retinaface', 'mediapipe', 'yolov8', 'yolov11s','yolov11n', 'yolov11m', 'yunet', 'centerface'

# Distance Metrics for Evaluation
DISTANCE_METRICS_TO_TEST = ["cosine", "euclidean", "euclidean_l2", 
# "manhattan", "mahalanobis", "hamming", "jaccard", "tanimoto", "pearson", "spearman", "kendall", "cosine_similarity", "euclidean_similarity", "manhattan_similarity", "mahalanobis_similarity", "hamming_similarity", "jaccard_similarity", "tanimoto_similarity", "pearson_similarity", "spearman_similarity", "kendall_similarity"
]

USE_SUPER_RESOLUTION = True

# Dataset Paths
DATASET_ROOT_DIR = "dataset/lfw-deepfunneled"  # Root for raw images. UPDATE THIS PATH
CROPPED_FACES_DIR = os.path.join(DATASET_ROOT_DIR, "1-cropped_faces") # Directory to store pre-cropped faces
FAISS_INDEXES_DIR = os.path.join(DATASET_ROOT_DIR, "0-faiss_indexes") # Directory to store faiss indexes

# Per-run Results Directory
RUNS_BASE_DIR = "results"

def get_run_folder_name():
    # Use timestamp and uuid for uniqueness
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"run_{timestamp}_{unique_id}"

RUN_FOLDER = os.path.join(RUNS_BASE_DIR, get_run_folder_name())

# Output Directories for this run
LOGS_DIR = RUN_FOLDER # os.path.join(RUN_FOLDER, "logs")
PLOTS_DIR = os.path.join(RUN_FOLDER, "plots")
REPORTS_DIR = os.path.join(RUN_FOLDER, "reports")

# Evaluation Parameters
THRESHOLD_PERCENTILE = 5

# Logging Configuration
LOG_FILENAME = os.path.join(LOGS_DIR, "evaluation_run.log")
LOG_LEVEL = "INFO" # e.g., INFO, DEBUG