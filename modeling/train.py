"""
HRV-based Cardiac Arrest Prediction Model Training Script
============================================================

Model training and evaluation pipeline for LightGBM model using SCOPE dataset

This script implements the complete model development pipeline for
predicting in-hospital cardiac arrest using heart rate variability (HRV)
features extracted from ICU patient ECG signals.

Model Architecture:
    - Algorithm: LightGBM (Light Gradient Boosting Machine)
    - Input: 33 HRV features selected via BorutaShap from initial 43 candidates
    - Output: Predicted probability of in-hospital cardiac arrest within 0.5–24h

Training Strategy:
    1. Stay-level train/test split to prevent data leakage
    2. Bayesian optimization with 5-fold stratified group cross-validation for hyperparameter tuning
    3. Early stopping with cross-validation to determine optimal boosting iterations
    4. Final model training on full development set with Beta calibration
    5. Hold-out test set evaluation

Output Files:
    - best_model.txt: Trained LightGBM model
    - calibrator.pkl: Fitted Beta calibration model
    - best_params.json: Optimized hyperparameters
    - training_results.json: Evaluation metrics with confidence intervals
    - predictions.csv: Sample-level predictions
    - plots/: Visualization outputs

Usage:
    python train.py \\
        --data_path input/scope_hrv_features.parquet \\
        --output_dir output/training_results \\
"""

import argparse
from bayes_opt import BayesianOptimization
from betacal import BetaCalibration
import lightgbm as lgb
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
import warnings
import json

from utils import (
    SEED,
    roc_ci, find_optimal_threshold, compute_metrics_at_threshold,
    plot_auc_prc_ci, time_plot_auc_prc_ci, 
    time_fine_cali, fine_cali, shap_plot
)

warnings.filterwarnings('ignore')


# ============================================================
# Constants
# ============================================================
# Bayesian Optimization settings
OPT_INITS = 3   # Number of random initialization points
OPT_ITERS = 50  # Number of optimization iterations per fold

# ============================================================
# HRV Feature Configuration
# ============================================================
# 33 HRV features selected via BorutaShap algorithm from initial 43 candidates.
#
# Categories include:
#   - Time-domain measures: MeanNN, SDNN, RMSSD, pNN50, pNN20, etc.
#   - Frequency-domain measures: VLF, LF, HF, VHF, LFn, HFn, LFHF
#   - Nonlinear measures: Poincaré (SD2, CSI, CSI_Modified),
#                         Fragmentation (PIP, IALS, PSS, PAS)
#
# HRVPARAMS follows the feature order in the trained model (bestmodel.txt).
HRVPARAMS = [
    'HRV_IALS', 'HRV_pNN50', 'HRV_TINN', 'HRV_HTI', 'HRV_PI', 'HRV_Prc20NN',
    'HRV_MedianNN', 'HRV_CVSD', 'HRV_PIP', 'HRV_MCVNN', 'HRV_LFn', 'HRV_CSI',
    'HRV_pNN20', 'HRV_MinNN', 'HRV_MeanNN', 'HRV_SDNNI1', 'HRV_MaxNN', 'HRV_HFn',
    'HRV_LF', 'HRV_LFHF', 'HRV_PAS', 'HRV_PSS', 'HRV_SDANN1', 'HRV_VHF',
    'HRV_RMSSD', 'HRV_VLF', 'HRV_IQRNN', 'HRV_CVNN', 'HRV_CSI_Modified',
    'HRV_SD2', 'HRV_Prc80NN', 'HRV_SD1SD2', 'HRV_SDNN',
]


# ============================================================
# Data Loading and Preprocessing
# ============================================================

def load_and_preprocess_data(data_path, feature_list=None):
    """
    Load SCOPE HRV features and preprocess for model training.
    
    This function prepares the SCOPE dataset for training, handling
    data loading, feature selection, and missing value removal.
    
    Expected parquet schema:
        - case_id: ICU stay identifier
        - label: Binary outcome (1=cardiac arrest, 0=no event)
        - time_to_event: Minutes until cardiac arrest (for time-stratified analysis)
        - HRV_*: 33 HRV features 
    
    Args:
        data_path: Path to preprocessed SCOPE HRV features (parquet format)
        feature_list: Optional list of specific features to use
    
    Returns:
        tuple: (DataFrame, list of feature names, case column name)
    
    Note:
        Stay-level splitting requires case_id column to prevent
        data leakage between train and test sets.
    """
    print("=" * 60)
    print("Loading and preprocessing data...")
    print("=" * 60)
    
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} samples from {data_path}")
    
    # Standardize column naming for compatibility with utils functions
    if 'time_to_event' in df.columns:
        df['time'] = df['time_to_event']
    
    # Handle infinite values from HRV calculations
    # (e.g., division by zero in ratio-based features)
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Determine features to use
    available_hrv = [c for c in df.columns if c.startswith('HRV_')]
    if feature_list:
        features = [f for f in feature_list if f in available_hrv]
        missing = [f for f in feature_list if f not in available_hrv]
        if missing:
            print(f"Warning: Missing features: {missing}")
    else:
        features = available_hrv
    
    print(f"\nUsing {len(features)} HRV features")
    
    # Remove samples with missing HRV values
    initial_len = len(df)
    df = df.dropna(subset=features)
    print(f"After removing NaN: {len(df)} samples (dropped {initial_len - len(df)})")
    
    # Identify case column for stay-level operations
    if 'case_id' in df.columns:
        case_col = 'case_id'
    elif 'stayid' in df.columns:
        case_col = 'stayid'
    else:
        case_col = None
        print("Warning: No case_id column found, cannot perform stay-level CV")
    
    # Dataset summary
    print(f"\nFinal data summary:")
    print(f"  Total samples: {len(df)}")
    print(f"  Unique stays: {df[case_col].nunique() if case_col else 'N/A'}")
    print(f"  Positive (label=1): {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")
    print(f"  Negative (label=0): {(df['label']==0).sum()}")
    
    return df, features, case_col


# ============================================================
# LightGBM Custom Metrics
# ============================================================

def lgb_auprc(preds, dtrain):
    """
    AUPRC evaluation metric for LightGBM.
    
    Args:
        preds: Model predictions
        dtrain: LightGBM Dataset with labels
    
    Returns:
        tuple: (metric_name, score, is_higher_better)
    """
    labels = dtrain.get_label()
    return 'auprc', average_precision_score(labels, preds), True


def lgb_auroc(preds, dtrain):
    """
    AUROC evaluation metric for LightGBM.
    
    Args:
        preds: Model predictions
        dtrain: LightGBM Dataset with labels
    
    Returns:
        tuple: (metric_name, score, is_higher_better)
    """
    labels = dtrain.get_label()
    return 'auroc', roc_auc_score(labels, preds), True


# ============================================================
# Model Training
# ============================================================

def train_model(df, features, case_col, output_dir, test_size=0.2, n_folds=5):
    """
    Complete training pipeline for HRV-based cardiac arrest prediction.
    
    This function implements the full model development workflow:
    
    1. Train/Test Split: Stay-level stratified split ensures no stay
       appears in both sets, preventing data leakage
    
    2. Bayesian Optimization: Efficient hyperparameter search using
       Gaussian process surrogate model, evaluated via 5-fold cross-validation
    
    3. Boosting Iteration Selection: Cross-validation determines optimal
       number of trees to prevent overfitting
    
    4. Final Model Training: Model trained on entire development set
       with Beta calibration for probability refinement
    
    5. Test Evaluation: Hold-out test set evaluation with bootstrap
       confidence intervals
    
    Args:
        df: Preprocessed DataFrame with HRV features and labels
        features: List of feature column names
        case_col: Column name for stay identifier
        output_dir: Directory for saving all outputs
        test_size: Fraction of stays for test set (default: 0.2)
        n_folds: Number of cross-validation folds (default: 5)
    
    Returns:
        tuple: (trained model, results dict, predictions DataFrame)
    
    Output Files:
        - best_model.txt: LightGBM model file
        - calibrator.pkl: Beta calibration model
        - best_params.json: Optimized hyperparameters
        - training_results.json: All evaluation metrics
        - predictions.csv: Sample-level predictions
    """
    print("\n" + "=" * 60)
    print("Starting model training...")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    plotdir = os.path.join(output_dir, 'plots')
    os.makedirs(plotdir, exist_ok=True)
    
    # ============================================================
    # 1. Train/Test Split (Stay-level)
    # ============================================================
    # Stay-level splitting prevents data leakage: epochs from the
    # same ICU stay cannot appear in both train and test sets
    print("\n[Step 1] Train/Test Split (Stay-level)")
    print("-" * 40)
    
    if case_col:
        unique_cases = df[case_col].unique()
        
        # Stratify by stay-level label (1 if any epoch is positive)
        case_labels = df.groupby(case_col)['label'].max().loc[unique_cases].values
        
        train_cases, test_cases = train_test_split(
            unique_cases, test_size=test_size, random_state=SEED, stratify=case_labels
        )
        
        train_df = df[df[case_col].isin(train_cases)].copy()
        test_df = df[df[case_col].isin(test_cases)].copy()
        
        print(f"Train stays: {len(train_cases)}, Test stays: {len(test_cases)}")
    else:
        # Fallback to sample-level split if no case id column
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=SEED, stratify=df['label']
        )
    
    print(f"Train samples: {len(train_df)} (pos: {train_df['label'].sum()})")
    print(f"Test samples: {len(test_df)} (pos: {test_df['label'].sum()})")
    
    # Prepare numpy arrays for training
    X_train = train_df[features].values.astype(float)
    y_train = train_df['label'].values.astype(int)
    X_test = test_df[features].values.astype(float)
    y_test = test_df['label'].values.astype(int)
    
    if case_col:
        groups_train = train_df[case_col].values
    
    # ============================================================
    # 2. Bayesian Optimization with 5-fold CV
    # ============================================================
    # Bayesian optimization efficiently searches hyperparameter space
    # by modeling performance as a Gaussian process
    print("\n[Step 2] Bayesian Optimization for Hyperparameters")
    print("-" * 40)
    
    # Fixed parameters (not optimized)
    fixed_params = {
        'objective': 'binary',
        'learning_rate': 0.005,
        'bagging_freq': 1,
        'force_row_wise': True,
        'max_depth': 5,
        'verbose': -1,
        'random_state': SEED,
        'n_jobs': 16,
    }
    
    # Parameter bounds for Bayesian optimization
    param_bounds = {
        'num_leaves': (16, 32),
        'lambda_l1': (0.7, 0.9),
        'lambda_l2': (0.9, 1),
        'feature_fraction': (0.6, 0.7),
        'bagging_fraction': (0.6, 0.9),
        'min_child_samples': (6, 10),
        'min_child_weight': (10, 40)
    }
    
    # Stratified Group K-Fold: stratifies by label while keeping
    # all epochs from same stay in same fold
    if case_col:
        folds = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    else:
        from sklearn.model_selection import StratifiedKFold
        folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    
    # Storage for fold data (accessed by eval_function)
    fold_data = {'X_valid': None, 'y_valid': None, 'dtrain': None, 'dvalid': None}
    
    def eval_function(num_leaves, lambda_l1, lambda_l2, feature_fraction,
                      bagging_fraction, min_child_samples, min_child_weight):
        """
        Objective function for Bayesian optimization.
        
        Trains LightGBM with given hyperparameters and returns AUPRC
        on validation fold. Early stopping prevents overfitting.
        """
        params = {
            'num_leaves': int(round(num_leaves)),
            'lambda_l1': lambda_l1,
            'lambda_l2': lambda_l2,
            'feature_fraction': feature_fraction,
            'bagging_fraction': bagging_fraction,
            'min_child_samples': int(round(min_child_samples)),
            'min_child_weight': min_child_weight,
            'feature_pre_filter': False,
        }
        params.update(fixed_params)
        
        model = lgb.train(
            params=params,
            train_set=fold_data['dtrain'],
            num_boost_round=2000,
            valid_sets=fold_data['dvalid'],
            feval=lgb_auprc,
            callbacks=[lgb.early_stopping(stopping_rounds=300, verbose=False)]
        )
        
        preds = model.predict(fold_data['X_valid'])
        score = average_precision_score(fold_data['y_valid'], preds)
        return score
    
    # Run optimization for each fold
    oof_max_params = []
    oof_max_scores = []
    
    split_args = (X_train, y_train, groups_train) if case_col else (X_train, y_train)
    
    for fold_idx, (train_idx, valid_idx) in enumerate(folds.split(*split_args)):
        print(f"\n### Fold {fold_idx + 1}/{n_folds} ###")
        
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_val, y_val = X_train[valid_idx], y_train[valid_idx]
        
        fold_data['X_valid'] = X_val
        fold_data['y_valid'] = y_val
        fold_data['dtrain'] = lgb.Dataset(X_tr, y_tr)
        fold_data['dvalid'] = lgb.Dataset(X_val, y_val)
        
        optimizer = BayesianOptimization(
            f=eval_function,
            pbounds=param_bounds,
            random_state=SEED,
            verbose=0
        )
        
        optimizer.maximize(init_points=OPT_INITS, n_iter=OPT_ITERS)
        
        oof_max_params.append(optimizer.max['params'])
        oof_max_scores.append(optimizer.max['target'])
        print(f"Best AUPRC: {optimizer.max['target']:.4f}")
    
    # Select parameters from fold with highest validation score
    best_fold_idx = np.argmax(oof_max_scores)
    best_params = oof_max_params[best_fold_idx].copy()
    best_params['num_leaves'] = int(round(best_params['num_leaves']))
    best_params['min_child_samples'] = int(round(best_params['min_child_samples']))
    best_params.update(fixed_params)
    
    print(f"\nBest params from fold {best_fold_idx + 1}: AUPRC = {oof_max_scores[best_fold_idx]:.4f}")
    print(f"Parameters: {best_params}")
    
    # Save optimized hyperparameters
    params_path = os.path.join(output_dir, 'best_params.json')
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"Saved parameters to {params_path}")
    
    # ============================================================
    # 3. Find Optimal Boosting Iterations
    # ============================================================
    # Cross-validation determines optimal number of boosting rounds
    # using early stopping, then averages across folds
    print("\n[Step 3] Finding Optimal Boosting Iterations")
    print("-" * 40)
    
    best_iters = []
    
    for fold_idx, (train_idx, valid_idx) in enumerate(folds.split(*split_args)):
        print(f"Fold {fold_idx + 1}/{n_folds}...", end=" ")
        
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_val, y_val = X_train[valid_idx], y_train[valid_idx]
        
        dtrain = lgb.Dataset(X_tr, y_tr)
        dvalid = lgb.Dataset(X_val, y_val)
        
        model = lgb.train(
            params=best_params,
            train_set=dtrain,
            num_boost_round=2000,
            valid_sets=dvalid,
            feval=lgb_auprc,
            callbacks=[lgb.early_stopping(stopping_rounds=300, verbose=False)]
        )
        
        best_iters.append(model.best_iteration)
        print(f"best_iter = {model.best_iteration}")
    
    final_iter = int(np.mean(best_iters))
    print(f"\nFinal boosting iterations: {final_iter} (mean of {best_iters})")
    
    # ============================================================
    # 4. Train Final Model and Calibration
    # ============================================================
    # Train on entire development set, then apply Beta calibration
    # to refine predicted probabilities
    print("\n[Step 4] Training Final Model and Calibration")
    print("-" * 40)
    
    dtrain_full = lgb.Dataset(X_train, y_train)
    
    final_model = lgb.train(
        params=best_params,
        train_set=dtrain_full,
        num_boost_round=final_iter
    )
    
    # Save trained model
    model_path = os.path.join(output_dir, 'best_model.txt')
    final_model.save_model(model_path)
    print(f"Saved model to {model_path}")
    
    # Beta Calibration
    # Refines probability estimates to improve calibration performance
    print("\nApplying Beta Calibration...")
    y_prob_train_raw = final_model.predict(X_train)
    
    calibrator = BetaCalibration()
    calibrator.fit(y_prob_train_raw.reshape(-1, 1), y_train)
    
    cal_path = os.path.join(output_dir, 'calibrator.pkl')
    with open(cal_path, 'wb') as f:
        pickle.dump(calibrator, f)
    print(f"Saved calibrator to {cal_path}")
    
    # ============================================================
    # 5. Evaluate on Test Set
    # ============================================================
    # Final evaluation on hold-out test set with calibrated predictions
    print("\n[Step 5] Evaluation on Test Set")
    print("-" * 40)
    
    # Generate raw predictions
    y_prob_train_raw = final_model.predict(X_train)
    y_prob_test_raw = final_model.predict(X_test)
    
    # Apply Beta calibration
    y_prob_train = calibrator.predict(y_prob_train_raw.reshape(-1, 1))
    y_prob_test = calibrator.predict(y_prob_test_raw.reshape(-1, 1))
    
    # Optimal threshold via Youden's J statistic
    thval = find_optimal_threshold(y_train, y_prob_train)
    print(f"Optimal threshold (from training): {thval:.4f}")
    
    # Discrimination metrics with bootstrap confidence intervals
    auroc = roc_auc_score(y_test, y_prob_test)
    auroc_lb, auroc_ub = roc_ci(y_test, y_prob_test)
    auprc = average_precision_score(y_test, y_prob_test)
    
    print(f"\n{'='*40}")
    print("TEST SET RESULTS")
    print(f"{'='*40}")
    print(f"AUROC: {auroc:.4f} (95% CI: {auroc_lb:.4f} - {auroc_ub:.4f})")
    print(f"AUPRC: {auprc:.4f}")
    
    # Classification metrics at optimal threshold
    metrics = compute_metrics_at_threshold(y_test, y_prob_test, thval)
    print(f"\nMetrics at threshold {thval:.4f}:")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  PPV: {metrics['ppv']:.4f}")
    print(f"  NPV: {metrics['npv']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    
    # ============================================================
    # 6. Save Results and Predictions
    # ============================================================
    print("\n[Step 6] Saving Results")
    print("-" * 40)
    
    # Add predictions to DataFrames (both raw and calibrated)
    train_df['y_prob_raw'] = y_prob_train_raw
    train_df['y_prob'] = y_prob_train
    train_df['y_true'] = y_train
    train_df['split'] = 'train'
    
    test_df['y_prob_raw'] = y_prob_test_raw
    test_df['y_prob'] = y_prob_test
    test_df['y_true'] = y_test
    test_df['split'] = 'test'
    
    # Save sample-level predictions
    result_df = pd.concat([train_df, test_df], ignore_index=True)
    pred_path = os.path.join(output_dir, 'predictions.csv')
    result_df.to_csv(pred_path, index=False)
    print(f"Saved predictions to {pred_path}")
    
    # Save results summary
    results = {
        # Primary discrimination metrics
        'auroc': float(auroc),
        'auroc_ci_lower': float(auroc_lb),
        'auroc_ci_upper': float(auroc_ub),
        'auprc': float(auprc),
        
        # Threshold and classification metrics
        'optimal_threshold': float(thval),
        'sensitivity': float(metrics['sensitivity']),
        'specificity': float(metrics['specificity']),
        'ppv': float(metrics['ppv']),
        'npv': float(metrics['npv']),
        'f1_score': float(metrics['f1_score']),
        'accuracy': float(metrics['accuracy']),
        
        # Training info
        'final_boosting_iterations': final_iter,
        'calibration_method': 'beta',
        
        # Dataset info
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'n_train_positive': int(y_train.sum()),
        'n_test_positive': int(y_test.sum()),
        'n_samples': len(X_test),  # for consistency with eval
        'n_positive': int(y_test.sum()),
        'n_negative': int((y_test == 0).sum()),
        'prevalence': float(np.mean(y_test)),
    }

    # ============================================================
    # 7. Generate Plots
    # ============================================================
    print("\n[Step 7] Generating Plots")
    print("-" * 40)
    
    try:
        # 1. Bootstrap AUROC/AUPRC curves
        print("\n1. Bootstrap AUROC/AUPRC curves...")
        bootstrap_results = plot_auc_prc_ci(y_test, y_prob_test, thval=thval, plotdir=plotdir)

        # Add bootstrap results to results dict
        results.update({
            'bootstrap_auroc': bootstrap_results['auroc'],
            'bootstrap_auroc_lower': bootstrap_results['auroc_lower'],
            'bootstrap_auroc_upper': bootstrap_results['auroc_upper'],
            'bootstrap_auprc': bootstrap_results['auprc'],
            'bootstrap_auprc_lower': bootstrap_results['auprc_lower'],
            'bootstrap_auprc_upper': bootstrap_results['auprc_upper'],
        })

        # 2. Time-stratified AUROC/AUPRC analysis
        if 'time' in test_df.columns:
            print("\n2. Time-stratified AUROC/AUPRC...")
            time_plot_auc_prc_ci(test_df, thval=thval, plotdir=plotdir)
            
            # 3. Time-stratified calibration
            print("\n3. Time-stratified calibration...")
            time_fine_cali(test_df, save_path=os.path.join(plotdir, 'calibration_by_time.png'))
        
        # 4. Overall calibration plot
        print("\n4. Calibration curve...")
        fine_cali(y_prob_test, y_test, save_path=os.path.join(plotdir, 'calibration.png'))

        # 5. SHAP feature importance analysis
        try:
            print("\n5. SHAP feature importance...")
            shap_plot(final_model, test_df[features].copy(), features, 
                    save_path=os.path.join(plotdir, 'shap_summary.png'))
        except Exception as e:
            print(f"  SHAP plot skipped: {e}")

    except Exception as e:
        print(f"Plot generation skipped: {e}")

    # Save results
    results_path = os.path.join(output_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_path}")
            
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model saved to: {model_path}")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)
    
    return final_model, results, result_df


# ============================================================
# Main Entry Point
# ============================================================

def main():
    """
    Command-line interface for model training.
    
    Example usage:
        python train.py \\
            --data_path input/scope_hrv_features.parquet \\
            --output_dir output/training_results \\
    """
    parser = argparse.ArgumentParser(description='Train HRV model on SCOPE data')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to SCOPE HRV features parquet file')
    parser.add_argument('--output_dir', type=str, default='./training_results',
                        help='Directory to save results')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Fraction of data for test set (default: 0.2)')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of CV folds (default: 5)')
    parser.add_argument('--feature_list', type=str, default=None,
                        help='Path to feature list JSON (optional)')
    
    args = parser.parse_args()
    
    # Load feature list if provided, otherwise use default HRVPARAMS
    feature_list = None
    if args.feature_list and os.path.exists(args.feature_list):
        with open(args.feature_list, 'r') as f:
            feature_list = json.load(f)
    else:
        feature_list = HRVPARAMS
    
    # Load and preprocess data
    df, features, case_col = load_and_preprocess_data(
        args.data_path,
        feature_list=feature_list
    )
    
    # Run training pipeline
    model, results, result_df = train_model(
        df=df,
        features=features,
        case_col=case_col,
        output_dir=args.output_dir,
        test_size=args.test_size,
        n_folds=args.n_folds
    )
    
    print("\nFinal Results Summary:")
    print(f"  AUROC: {results['auroc']:.4f} ({results['auroc_ci_lower']:.4f}-{results['auroc_ci_upper']:.4f})")
    print(f"  AUPRC: {results['auprc']:.4f}")


if __name__ == '__main__':
    main()
