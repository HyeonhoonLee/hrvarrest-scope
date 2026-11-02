"""
HRV-based Cardiac Arrest Prediction Model Evaluation Script
============================================================

Inference and evaluation pipeline for the trained LightGBM model

This script performs inference and evaluation on the final HRV-based
cardiac arrest prediction model, including discrimination, calibration,
and time-stratified analyses.

Model Architecture:
    - Algorithm: LightGBM
    - Input: 33 HRV features selected via BorutaShap from initial 43 candidates
    - Output: Predicted probability of in-hospital cardiac arrest within 0.5–24h

Evaluation Components:
    - Discrimination metrics: AUROC, AUPRC, sensitivity,
      specificity, PPV, accuracy, and F1-score
    - Calibration analysis: agreement between predicted
      probabilities and observed event rates
    - Time-stratified analysis: performance across multiple
      prediction horizons

Statistical Methods:
    - Bootstrap resampling for confidence interval estimation
    - Optimal classification threshold selected via
      Youden's J statistic

Pipeline Summary:
    1. Load preprocessed HRV features
    2. Align feature order with trained model's expected input
    3. Generate prediction probabilities using trained LightGBM model
    4. Compute performance metrics with bootstrap confidence intervals
    5. Perform time-stratified evaluation
    6. Save results and visualization outputs

Note:
    - This script performs inference only (no model training)
    - The model file (.txt) contains feature names and order
    - Feature order must exactly match the training configuration

Usage:
    python evaluation.py \\
        --data_path input/scope_hrv_features.parquet \\
        --model_path input/model.txt \\
        --output_dir output/evaluation_results
"""

import pandas as pd
import numpy as np
import os
import argparse
import json
import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

from utils import (
    roc_ci, find_optimal_threshold, compute_metrics_at_threshold,
    plot_auc_prc_ci, time_plot_auc_prc_ci,
    fine_cali, time_fine_cali, shap_plot
)


# ============================================================
# HRV Feature Configuration
# ============================================================
# 33 HRV features selected via BorutaShap algorithm from initial 43 candidates.
# Only features classified as 'accepted' by BorutaShap were included in model development.
#
# Categories include:
#   - Time-domain measures: MeanNN, SDNN, RMSSD, pNN50, pNN20, etc.
#   - Frequency-domain measures: VLF, LF, HF, VHF, LFn, HFn, LFHF
#   - Nonlinear measures: Poincaré (SD2, CSI, CSI_Modified), 
#                         Fragmentation (PIP, IALS, PSS, PAS)
#
# HRVPARAMS is pre-defined in the same order as the trained model (bestmodel.txt).
HRVPARAMS = [
    'HRV_IALS', 'HRV_pNN50', 'HRV_TINN', 'HRV_HTI', 'HRV_PI', 'HRV_Prc20NN',
    'HRV_MedianNN', 'HRV_CVSD', 'HRV_PIP', 'HRV_MCVNN', 'HRV_LFn', 'HRV_CSI',
    'HRV_pNN20', 'HRV_MinNN', 'HRV_MeanNN', 'HRV_SDNNI1', 'HRV_MaxNN', 'HRV_HFn',
    'HRV_LF', 'HRV_LFHF', 'HRV_PAS', 'HRV_PSS', 'HRV_SDANN1', 'HRV_VHF',
    'HRV_RMSSD', 'HRV_VLF', 'HRV_IQRNN', 'HRV_CVNN', 'HRV_CSI_Modified',
    'HRV_SD2', 'HRV_Prc80NN', 'HRV_SD1SD2', 'HRV_SDNN',
]


# ============================================================
# Data Loading Functions
# ============================================================

def get_model_features(model_path):
    """
    Extract feature names directly from LightGBM model file.
    
    LightGBM saves feature names in the model file header as:
        'feature_names=feature1 feature2 feature3 ...'
    
    This ensures feature alignment between the trained model
    and the inference data, preventing column mismatch errors.
    
    Args:
        model_path: Path to the LightGBM model file (.txt format)
    
    Returns:
        list: Feature names in the order expected by the model,
              or None if feature_names line is not found
    
    Note:
        Feature order is critical for LightGBM - predictions will be
        incorrect if features are provided in a different order than
        during training.
    """
    with open(model_path, 'r') as f:
        for line in f:
            if line.startswith('feature_names='):
                features = line.strip().replace('feature_names=', '').split(' ')
                return features
    return None


def load_and_preprocess_data(data_path, model_path=None):
    """
    Load SCOPE HRV features and preprocess for model evaluation.
    
    This function prepares the SCOPE dataset for inference using 
    the trained model. It handles data loading, feature verification, 
    and missing value removal.
    
    Expected parquet schema:
        - case_id: ICU stay identifier for stay-level analysis
        - label: Binary outcome (1=cardiac arrest, 0=no event)
        - time_to_event: Minutes until cardiac arrest (for time-stratified analysis)
                        Used to define secondary outcomes
        - quality_ratio: ECG signal quality metric from preprocessing
        - HRV_*: 33 HRV features extracted from 5-minute ECG segments
    
    Args:
        data_path: Path to preprocessed SCOPE HRV features (parquet format)
        model_path: Path to trained model file for feature name extraction
    
    Returns:
        tuple: (DataFrame with processed data, list of feature column names)
    
    Raises:
        ValueError: If required model features are missing from the dataset
    
    Note:
        The function prioritizes features from the model file to ensure exact alignment. 
        Samples with any NaN values in HRV features are dropped 
        as LightGBM cannot handle missing values in this configuration.
    """
    print("=" * 60)
    print("Loading and preprocessing data...")
    print("=" * 60)
    
    # Load preprocessed HRV features from SCOPE dataset
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} samples from {data_path}")
    print(f"Columns ({len(df.columns)}): {df.columns.tolist()[:10]}...")
    
    # Standardize column naming for compatibility with utils functions
    # 'time_to_event' -> 'time' for time-stratified analysis functions
    if 'time_to_event' in df.columns:
        df['time'] = df['time_to_event']
    
    # Handle infinite values that may result from HRV calculations
    # (e.g., division by zero in ratio-based features like LFHF, SD1SD2)
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Determine required features: prioritize model's feature list for alignment
    # This ensures inference uses exactly the features the model was trained on
    model_features = get_model_features(model_path) if model_path else None
    required_features = model_features or HRVPARAMS
    
    # Verify feature availability in the SCOPE dataset
    hrv_features = [c for c in df.columns if c.startswith('HRV_')]
    available_features = [f for f in required_features if f in hrv_features]
    missing_features = [f for f in required_features if f not in hrv_features]
    
    print(f"\nFeature availability: {len(available_features)}/{len(required_features)}")
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")
    
    # Remove samples with missing HRV values
    # LightGBM requires complete feature vectors for prediction
    initial_len = len(df)
    df = df.dropna(subset=available_features)
    print(f"\nAfter removing NaN: {len(df)} samples (dropped {initial_len - len(df)})")
    
    # Dataset summary for reporting
    print(f"\nData summary:")
    print(f"  Total samples: {len(df)}")
    print(f"  Features: {len(required_features)}")
    print(f"  Positive (label=1): {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")
    print(f"  Negative (label=0): {(df['label']==0).sum()} ({(df['label']==0).mean()*100:.1f}%)")
    
    return df, required_features


# ============================================================
# Model Evaluation Function
# ============================================================

def evaluate_model(model_path, df, feature_cols, output_dir='./evaluation_results'):
    """
    Evaluate trained LightGBM model on SCOPE dataset.
    
    This function implements the complete evaluation pipeline:
    
    1. Discrimination metrics: AUROC, AUPRC, sensitivity, specificity, 
       PPV, NPV, accuracy, F1-score with bootstrap 95% CI
    
    2. Optimal threshold selection: Youden's J statistic maximization
       Threshold = argmax(Sensitivity + Specificity - 1)
    
    3. Time-stratified analysis: Performance across prediction horizons
       Evaluates model at secondary outcomes (0.5–18h, 12h, 6h, 3h, 1h)
    
    4. Calibration assessment: Predicted vs observed probability comparison
       Compares predicted probabilities against observed fractions of cardiac arrest
    
    5. Feature importance: SHAP-based interpretation
       Quantifies each feature's contribution to prediction using SHAP values
    
    Args:
        model_path: Path to trained LightGBM model (.txt format)
        df: Preprocessed DataFrame with HRV features and labels
        feature_cols: List of feature column names
        output_dir: Directory for saving results and plots
    
    Returns:
        tuple: (results_dict with all metrics, DataFrame with predictions)
    
    Output Files:
        - evaluation_results.json: All computed metrics with CIs
        - predictions.csv: Sample-level predictions for downstream analysis
        - plots/: Directory containing visualization figures
            - auroc_ci.png: ROC curves with bootstrap CI
            - auroc_by_time.png: Time-stratified AUROC analysis
            - auprc_ci.png: PR curves with bootstrap CI
            - auprc_by_time.png: Time-stratified AUPRC analysis            
            - calibration_by_time.png: Time-stratified calibration curves
            - calibration.png: Overall calibration plot
            - shap_summary.png: SHAP feature importance
    """
    print("\n" + "=" * 60)
    print("Evaluating model...")
    print("=" * 60)
    
    # Initialize output directories
    os.makedirs(output_dir, exist_ok=True)
    plotdir = os.path.join(output_dir, 'plots')
    os.makedirs(plotdir, exist_ok=True)
    
    # --------------------------------------------------------
    # Load Trained Model
    # --------------------------------------------------------
    # Final LightGBM model trained on entire development set
    print(f"Loading model from {model_path}")
    model = lgb.Booster(model_file=model_path)
    
    # --------------------------------------------------------
    # Feature Alignment Verification
    # --------------------------------------------------------
    # Critical: Features must be in the exact order used during training
    model_features = model.feature_name()
    print(f"\nModel expects {len(model_features)} features: {model_features[:5]}...")
    print(f"Data prepared with {len(feature_cols)} features")
    
    # Verify all required features exist in the dataset
    missing = [f for f in model_features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features in data: {missing}")
    
    # Reorder columns to match model's expected feature order
    X = df[model_features].copy()
    y = df['label'].values
    
    # --------------------------------------------------------
    # Generate Predictions
    # --------------------------------------------------------
    # Model outputs probability of cardiac arrest within prediction window
    # Primary outcome: cardiac arrest within 0.5-24 hours
    print("\nMaking predictions...")
    y_prob = model.predict(X)
    
    # Store predictions in DataFrame for time-stratified analysis
    df['y_prob'] = y_prob
    df['y_true'] = y
    
    # ============================================================
    # Compute Discrimination Metrics
    # ============================================================
    print("\n" + "-" * 40)
    print("EVALUATION RESULTS")
    print("-" * 40)
    
    # AUROC: Area Under Receiver Operating Characteristic curve
    auroc = roc_auc_score(y, y_prob)
    auroc_lower, auroc_upper = roc_ci(y, y_prob)
    print(f"\nAUROC: {auroc:.4f} (95% CI: {auroc_lower:.4f} - {auroc_upper:.4f})")
    
    # AUPRC: Area Under Precision-Recall Curve
    auprc = average_precision_score(y, y_prob)
    print(f"AUPRC: {auprc:.4f}")
    
    # --------------------------------------------------------
    # Optimal Threshold Selection
    # --------------------------------------------------------
    # Youden's J statistic: maximizes (Sensitivity + Specificity - 1)
    thval = find_optimal_threshold(y, y_prob)
    print(f"\nOptimal threshold (Youden's J): {thval:.4f}")
    
    # --------------------------------------------------------
    # Classification Metrics at Optimal Threshold
    # --------------------------------------------------------
    metrics = compute_metrics_at_threshold(y, y_prob, thval)
    print(f"\nMetrics at optimal threshold:")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")     # True Positive Rate (recall)
    print(f"  Specificity: {metrics['specificity']:.4f}")     # True Negative Rate
    print(f"  PPV: {metrics['ppv']:.4f}")                     # Precision (Positive Predictive Value)
    print(f"  NPV: {metrics['npv']:.4f}")                     # Negative Predictive Value
    print(f"  F1 Score: {metrics['f1_score']:.4f}")           # Harmonic mean of precision and recall
    print(f"  Accuracy: {metrics['accuracy']:.4f}")           # Overall classification accuracy
    print(f"\nConfusion Matrix: TP={metrics['tp']}, FP={metrics['fp']}, TN={metrics['tn']}, FN={metrics['fn']}")
    
    # ============================================================
    # Generate Visualization Plots
    # ============================================================
    print("\n" + "-" * 40)
    print("Generating plots...")
    print("-" * 40)
    
    # 1. Bootstrap AUROC/AUPRC curves with 95% confidence intervals
    print("\n1. Bootstrap AUROC/AUPRC curves...")
    bootstrap_results = plot_auc_prc_ci(y, y_prob, thval=thval, plotdir=plotdir)
    
    # 2. Time-stratified performance analysis
    # Secondary outcomes analysis across different prediction horizons
    if 'time' in df.columns:
        print("\n2. Time-stratified AUROC/AUPRC...")
        time_results = time_plot_auc_prc_ci(df, thval=thval, plotdir=plotdir)
        
        # 3. Calibration by prediction horizon
        print("\n3. Time-stratified calibration...")
        time_fine_cali(df, save_path=os.path.join(plotdir, 'calibration_by_time.png'))
    
    # --------------------------------------------------------
    # Additional Analyses
    # --------------------------------------------------------
    # 4. Overall calibration plot
    print("\n4. Calibration curve...")
    fine_cali(y_prob, y, save_path=os.path.join(plotdir, 'calibration.png'))
    
    # 5. SHAP feature importance analysis
    try:
        print("\n5. SHAP feature importance...")
        shap_plot(model, X, model_features, 
                  save_path=os.path.join(plotdir, 'shap_summary.png'))
    except Exception as e:
        print(f"  SHAP plot skipped: {e}")
    
    # ============================================================
    # Export Results
    # ============================================================
    
    # Compile all metrics into results dictionary
    results = {
        # Primary discrimination metrics
        'auroc': float(auroc),
        'auroc_ci_lower': float(auroc_lower),
        'auroc_ci_upper': float(auroc_upper),
        'auprc': float(auprc),
        
        # Threshold and classification metrics
        'optimal_threshold': float(thval),
        'sensitivity': float(metrics['sensitivity']),
        'specificity': float(metrics['specificity']),
        'ppv': float(metrics['ppv']),
        'npv': float(metrics['npv']),
        'f1_score': float(metrics['f1_score']),
        'accuracy': float(metrics['accuracy']),
        
        # Dataset characteristics for reporting
        'n_samples': len(y),
        'n_positive': int(np.sum(y)),
        'n_negative': int(np.sum(y == 0)),
        'prevalence': float(np.mean(y))
    }
    
    # Add bootstrap confidence interval results
    results.update({
        'bootstrap_auroc': bootstrap_results['auroc'],
        'bootstrap_auroc_lower': bootstrap_results['auroc_lower'],
        'bootstrap_auroc_upper': bootstrap_results['auroc_upper'],
        'bootstrap_auprc': bootstrap_results['auprc'],
        'bootstrap_auprc_lower': bootstrap_results['auprc_lower'],
        'bootstrap_auprc_upper': bootstrap_results['auprc_upper'],
    })
    
    # Save evaluation metrics to JSON for documentation
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Save sample-level predictions for downstream analysis
    # Includes probability scores and binary predictions at optimal threshold
    pred_df = df[['case_id', 'time', 'y_true', 'y_prob']].copy() if 'case_id' in df.columns else \
              df[['time', 'y_true', 'y_prob']].copy()
    pred_df['y_pred'] = (y_prob >= thval).astype(int)
    pred_path = os.path.join(output_dir, 'predictions.csv')
    pred_df.to_csv(pred_path, index=False)
    print(f"Predictions saved to {pred_path}")
    
    return results, df


# ============================================================
# Main Entry Point
# ============================================================

def main():
    """
    Command-line interface for model evaluation.
    
    Example usage:
        python evaluation.py \\
            --data_path input/scope_hrv_features.parquet \\
            --model_path input/model.txt \\
            --output_dir output/evaluation_results
    """
    parser = argparse.ArgumentParser(description='Evaluate HRV model on SCOPE data')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to SCOPE HRV features parquet file')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained LightGBM model (.txt)')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Step 1: Load and preprocess SCOPE dataset
    # Pass model_path to extract expected feature list from model file
    df, feature_cols = load_and_preprocess_data(
        args.data_path,
        model_path=args.model_path,
    )
    
    # Step 2: Run model evaluation pipeline
    results, result_df = evaluate_model(
        args.model_path,
        df,
        feature_cols,
        args.output_dir
    )
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
