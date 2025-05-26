# reporting/results_visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import logging
import numpy as np # Import numpy for isnan checks

from config import settings
from common import utils

logger = logging.getLogger(__name__)

class ResultsVisualizer:
    def __init__(self,
                 plots_dir: str = settings.PLOTS_DIR,
                 reports_dir: str = settings.REPORTS_DIR):
        self.plots_dir = plots_dir
        self.reports_dir = reports_dir
        utils.ensure_dir_exists(self.plots_dir)
        utils.ensure_dir_exists(self.reports_dir)

    def setup_logging(self, log_file: str = settings.LOG_FILENAME, level: str = settings.LOG_LEVEL):
        utils.ensure_dir_exists(os.path.dirname(log_file))
        log_level_enum = getattr(logging, level.upper(), logging.INFO)
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(
            level=log_level_enum,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w'),
                logging.StreamHandler()
            ]
        )
        logger.info(f"Logging configured. Level: {level}. File: {log_file}")

    def log_experiment_summary(self, model_name: str, detector: str, metric: str,
                               accuracy_results: dict, avg_embedding_time_ms: float,
                               evaluation_duration_s: float, num_embeddings: int):
        # Ensure accuracy_results values are serializable for logging if they are numpy types
        threshold_val = accuracy_results.get('threshold', 'N/A')
        if isinstance(threshold_val, (np.float32, np.float64)):
            threshold_val = float(threshold_val) if not np.isnan(threshold_val) else 'N/A'
        
        accuracy_val = accuracy_results.get('accuracy', 'N/A')
        if isinstance(accuracy_val, (np.float32, np.float64)):
            accuracy_val = float(accuracy_val) if not np.isnan(accuracy_val) else 'N/A'
            
        precision_val = accuracy_results.get('precision', 'N/A')
        if isinstance(precision_val, (np.float32, np.float64)):
            precision_val = float(precision_val) if not np.isnan(precision_val) else 'N/A'

        recall_val = accuracy_results.get('recall_tpr', 'N/A')
        if isinstance(recall_val, (np.float32, np.float64)):
            recall_val = float(recall_val) if not np.isnan(recall_val) else 'N/A'

        fpr_val = accuracy_results.get('fpr', 'N/A')
        if isinstance(fpr_val, (np.float32, np.float64)):
            fpr_val = float(fpr_val) if not np.isnan(fpr_val) else 'N/A'


        log_message = (
            f"Experiment Summary:\n"
            f"\tModel: {model_name}\n"
            f"\tCropping Detector: {detector}\n" # Changed label for clarity
            f"\tDistance Metric: {metric}\n"
            f"\tNumber of Embeddings: {num_embeddings}\n"
            f"\tAvg. Embedding Time (ms/image): {avg_embedding_time_ms:.2f}\n"
            f"\tEvaluation Duration (s): {evaluation_duration_s:.2f}\n"
            f"\tThreshold (@{settings.THRESHOLD_PERCENTILE}th percentile of neg dists): {threshold_val if threshold_val == 'N/A' else f'{threshold_val:.4f}'}\n"
            f"\tAccuracy: {accuracy_val if accuracy_val == 'N/A' else f'{accuracy_val:.4f}'}\n"
            f"\tPrecision: {precision_val if precision_val == 'N/A' else f'{precision_val:.4f}'}\n"
            f"\tRecall (TPR): {recall_val if recall_val == 'N/A' else f'{recall_val:.4f}'}\n"
            f"\tFPR: {fpr_val if fpr_val == 'N/A' else f'{fpr_val:.4f}'}\n"
            f"\tTP: {accuracy_results.get('true_positives', 'N/A')}, FN: {accuracy_results.get('false_negatives', 'N/A')}, "
            f"TN: {accuracy_results.get('true_negatives', 'N/A')}, FP: {accuracy_results.get('false_positives', 'N/A')}"
        )
        logger.info(log_message)


    def plot_distance_distributions(self,
                                    positive_distances: list,
                                    negative_distances: list,
                                    model_name: str,
                                    distance_metric: str,
                                    threshold: float,
                                    filename_suffix: str = ""):
        if not positive_distances and not negative_distances:
            logger.warning(f"No distances to plot for {model_name} - {distance_metric}.")
            return

        plt.figure(figsize=(12, 7))
        if positive_distances:
            sns.histplot(positive_distances, color="blue", label=f'Same Identity (N={len(positive_distances)})', kde=True, stat="density", element="step")
        if negative_distances:
            sns.histplot(negative_distances, color="red", label=f'Different Identities (N={len(negative_distances)})', kde=True, stat="density", element="step")

        if threshold is not None and not (isinstance(threshold, str) and threshold == "N/A") and not np.isnan(threshold) :
            plt.axvline(threshold, color='green', linestyle='dashed', linewidth=2, label=f'Threshold = {threshold:.4f}')

        plt.title(f'Distance Distribution: {model_name} ({distance_metric})')
        plt.xlabel(f'{distance_metric.capitalize()} Distance')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        plot_filename = f"{model_name}_{distance_metric}_{filename_suffix}_distances.png".replace('-', '_').lower()
        plot_path = os.path.join(self.plots_dir, plot_filename)
        try:
            plt.savefig(plot_path)
            logger.info(f"Saved distance distribution plot to {plot_path}")
        except Exception as e:
            logger.error(f"Failed to save plot {plot_path}: {e}")
        plt.close()

    def generate_summary_report_table(self, all_results: list, filename: str = "summary_evaluation_report.csv"):
        if not all_results:
            logger.warning("No results to generate a summary report.")
            return

        df = pd.DataFrame(all_results)
        report_path = os.path.join(self.reports_dir, filename)
        try:
            df.to_csv(report_path, index=False)
            logger.info(f"Summary report saved to {report_path}")
        except Exception as e:
            logger.error(f"Failed to save summary report {report_path}: {e}")

        logger.info("\n" + "="*20 + " Overall Evaluation Summary " + "="*20)
        
        # Define expected columns, including the renamed 'cropping_detector'
        expected_columns = [
            "model", "cropping_detector", "distance_metric", "num_embeddings",
            "accuracy", "threshold", "precision", "recall_tpr", "fpr",
            "avg_embedding_time_ms", "evaluation_time_s"
        ]
        
        # Filter DataFrame to include only expected columns that are present
        # This prevents KeyErrors if a column is entirely missing from all_results
        display_columns = [col for col in expected_columns if col in df.columns]
        if not display_columns:
            logger.error("No valid columns found for the summary report. DataFrame columns: %s", df.columns.tolist())
            return

        display_df = df[display_columns].copy()

        # Columns for numeric conversion and rounding
        numeric_cols_4_decimals = ['accuracy', 'threshold', 'precision', 'recall_tpr', 'fpr']
        numeric_cols_2_decimals = ['avg_embedding_time_ms', 'evaluation_time_s']

        for col in numeric_cols_4_decimals:
            if col in display_df.columns:
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce').round(4)
        
        for col in numeric_cols_2_decimals:
            if col in display_df.columns:
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce').round(2)
        
        try:
            from tabulate import tabulate
            logger.info("\n" + tabulate(display_df, headers='keys', tablefmt='psql', showindex=False))
        except ImportError:
            logger.info("\n" + display_df.to_string(index=False))
        logger.info("="* (40 + len(" Overall Evaluation Summary ")) + "\n")

