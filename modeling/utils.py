"""
Utility Functions for HRV-based Cardiac Arrest Prediction Model Evaluation
===========================================================================

Statistical analysis, visualization, and evaluation utilities for the 
cardiac arrest prediction model

Module Structure:
    1. Configuration Constants
    2. Confidence Interval Functions
    3. Basic Metric Functions
    4. Bootstrap Evaluation Functions
    5. Calibration Functions
    6. SHAP Functions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from math import ceil
import scipy.stats as stat
from scipy.stats import beta, percentileofscore

from sklearn.metrics import (
    roc_auc_score, 
    roc_curve, 
    auc, 
    precision_recall_curve, 
    average_precision_score,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve

import os


# ============================================================
# Configuration Constants
# ============================================================
# Time threshold defining the prediction window boundary
CRITERIA = 24 * 60  # 24 hours in minutes

SEED = 42
NUM_BOOTSTRAPS = 2000


# ============================================================
# Confidence Interval Functions
# ============================================================

def score_ci(
    y_true,
    y_pred,
    score_fun,
    sample_weight=None,
    n_bootstraps=2000,
    confidence_level=0.95,
    seed=None,
    reject_one_class_samples=True,
):
    """
    Compute confidence interval for a score function using bootstrap resampling.
    
    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted probabilities
        score_fun: Scoring function (e.g., roc_auc_score)
        n_bootstraps: Number of bootstrap iterations
        confidence_level: Confidence level (default 0.95 for 95% CI)
        seed: Random seed for reproducibility
        reject_one_class_samples: Skip bootstrap samples with only one class
    
    Returns:
        tuple: (score, ci_lower, ci_upper, scores)
    """
    assert len(y_true) == len(y_pred)

    score = score_fun(y_true, y_pred)
    _, ci_lower, ci_upper, scores = score_stat_ci(
        y_true=y_true,
        y_preds=y_pred,
        score_fun=score_fun,
        sample_weight=sample_weight,
        n_bootstraps=n_bootstraps,
        confidence_level=confidence_level,
        seed=seed,
        reject_one_class_samples=reject_one_class_samples,
    )

    return score, ci_lower, ci_upper, scores


def score_stat_ci(
    y_true,
    y_preds,
    score_fun,
    stat_fun=np.mean,
    sample_weight=None,
    n_bootstraps=2000,
    confidence_level=0.95,
    seed=None,
    reject_one_class_samples=True,
):
    """
    Compute confidence interval for a statistic of score function using bootstrap.
    
    Supports multiple prediction sets for ensemble or multi-reader analysis.
    
    Args:
        y_true: Ground truth binary labels
        y_preds: Predicted probabilities (can be 2D for multiple readers)
        score_fun: Scoring function
        stat_fun: Aggregation function (default: np.mean)
        n_bootstraps: Number of bootstrap iterations
        confidence_level: Confidence level for the interval
    
    Returns:
        tuple: (mean_score, ci_lower, ci_upper, scores)
    """
    y_true = np.array(y_true)
    y_preds = np.atleast_2d(y_preds)
    assert all(len(y_true) == len(y) for y in y_preds)

    np.random.seed(seed)
    scores = []
    
    for i in range(n_bootstraps):
        readers = np.random.randint(0, len(y_preds), len(y_preds))
        indices = np.random.randint(0, len(y_true), len(y_true))
        
        if reject_one_class_samples and len(np.unique(y_true[indices])) < 2:
            continue
        
        reader_scores = []
        for r in readers:
            if sample_weight is not None:
                reader_scores.append(
                    score_fun(y_true[indices], y_preds[r][indices], 
                              sample_weight=sample_weight[indices])
                )
            else:
                sorted_indices = np.sort(indices)[::-1]
                reader_scores.append(
                    score_fun(y_true[sorted_indices], y_preds[r][sorted_indices])
                )
        
        scores.append(stat_fun(reader_scores))

    mean_score = np.mean(scores)
    sorted_scores = np.array(sorted(scores))
    
    alpha = (1.0 - confidence_level) / 2.0
    ci_lower = sorted_scores[int(round(alpha * len(sorted_scores)))]
    ci_upper = sorted_scores[int(round((1.0 - alpha) * len(sorted_scores)))]
    
    return mean_score, ci_lower, ci_upper, scores


def pvalue(
    y_true,
    y_pred1,
    y_pred2,
    score_fun,
    sample_weight=None,
    n_bootstraps=2000,
    two_tailed=True,
    seed=None,
    reject_one_class_samples=True,
):
    """
    Compute p-value for comparing two models using bootstrap hypothesis testing.
    
    Args:
        y_true: Ground truth labels
        y_pred1: Predictions from Model 1
        y_pred2: Predictions from Model 2
        score_fun: Scoring function (e.g., roc_auc_score)
        sample_weight: Sample weights for weighted metrics (optional)
        n_bootstraps: Number of bootstrap iterations (default: 2000)
        two_tailed: If True, compute two-tailed p-value (default: True)
        seed: Random seed for reproducibility (optional)
        reject_one_class_samples: Skip bootstrap samples with only one class (default: True)
    
    Returns:
        tuple: (p_value, z_scores)
    """
    assert len(y_true) == len(y_pred1) == len(y_pred2)

    y_true = np.array(y_true)
    y_preds1 = np.atleast_2d(y_pred1)
    y_preds2 = np.atleast_2d(y_pred2)

    np.random.seed(seed)
    z = []
    
    for i in range(n_bootstraps):
        readers1 = np.random.randint(0, len(y_preds1), len(y_preds1))
        readers2 = np.random.randint(0, len(y_preds2), len(y_preds2))
        indices = np.random.randint(0, len(y_true), len(y_true))
        
        if reject_one_class_samples and len(np.unique(y_true[indices])) < 2:
            continue
        
        reader1_scores = []
        for r in readers1:
            if sample_weight is not None:
                reader1_scores.append(
                    score_fun(y_true[indices], y_preds1[r][indices], 
                              sample_weight=sample_weight[indices])
                )
            else:
                reader1_scores.append(
                    score_fun(y_true[indices], y_preds1[r][indices])
                )
        score1 = np.mean(reader1_scores)
        
        reader2_scores = []
        for r in readers2:
            if sample_weight is not None:
                reader2_scores.append(
                    score_fun(y_true[indices], y_preds2[r][indices], 
                              sample_weight=sample_weight[indices])
                )
            else:
                reader2_scores.append(
                    score_fun(y_true[indices], y_preds2[r][indices])
                )
        score2 = np.mean(reader2_scores)
        
        z.append(score1 - score2)

    p = percentileofscore(z, 0.0, kind="weak") / 100.0
    if two_tailed:
        p *= 2.0
    
    return p, z


# ============================================================
# Basic Metric Functions
# ============================================================

def roc_ci(y_true, y_pred):
    """
    Calculate AUROC confidence interval using beta distribution approximation.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
    
    Returns:
        tuple: (lower, upper) bounds of 95% CI
    """
    total = len(y_true)
    success = roc_auc_score(y_true, y_pred) * total
    alpha = 0.05
    
    lower = stat.beta.ppf(alpha / 2, success, total - success + 1)
    upper = stat.beta.ppf(1 - alpha / 2, success + 1, total - success)
    
    return lower, upper


def find_optimal_threshold(y_true, y_prob):
    """
    Find optimal classification threshold using Youden's J statistic.
    
    J = Sensitivity + Specificity - 1 = TPR - FPR
    
    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
    
    Returns:
        float: Optimal threshold value
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    return thresholds[optimal_idx]


def compute_metrics_at_threshold(y_true, y_prob, threshold):
    """
    Compute classification metrics at a given threshold.
    
    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        threshold: Classification threshold
    
    Returns:
        dict: sensitivity, specificity, ppv, npv, f1_score, accuracy, 
              confusion matrix components
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    acc = (tp + tn) / (tn + fp + fn + tp)
    
    return {
        'threshold': threshold,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'f1_score': f1,
        'accuracy': acc,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
    }


# ============================================================
# Bootstrap Evaluation Functions
# ============================================================

def plot_auc_prc_ci(y_true, y_prob, thval=None, num_bootstraps=2000, plotdir='./plots'):
    """
    Generate AUROC and AUPRC curves with bootstrap 95% confidence intervals.
    
    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        thval: Classification threshold (if None, uses Youden's J)
        num_bootstraps: Number of bootstrap iterations
        plotdir: Directory for output plots
    
    Returns:
        dict: AUROC, AUPRC with CIs, and other metrics
    
    Output Files:
        - auroc_ci.png: ROC curve with confidence band
        - auprc_ci.png: PR curve with confidence band
    """
    os.makedirs(plotdir, exist_ok=True)
    
    tprs, aucs = [], []
    auprcs, precision_arrs = [], []
    sens, accs, spes, pres, f1s = [], [], [], [], []
    
    mean_fpr = np.linspace(0, 1, 100)
    mean_recall = np.linspace(0, 1, 100)
    
    y_true, y_prob = np.array(y_true), np.array(y_prob)
    
    # Find optimal threshold if not provided
    if thval is None:
        thval = find_optimal_threshold(y_true, y_prob)
    
    for _ in range(num_bootstraps):
        idx = np.random.choice(range(len(y_true)), size=len(y_true), replace=True)
        
        if len(np.unique(y_true[idx])) < 2:
            continue
        
        # ROC
        fpr, tpr, _ = roc_curve(y_true[idx], y_prob[idx])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(auc(fpr, tpr))
        
        # PR
        precision, recall, _ = precision_recall_curve(y_true[idx], y_prob[idx])
        precision_arr = np.interp(mean_recall, recall[::-1], precision[::-1])
        auprcs.append(auc(mean_recall, precision_arr))
        precision_arrs.append(precision_arr)
        
        # Threshold-based metrics
        y_pred_binary = y_prob[idx] > thval
        f1s.append(f1_score(y_true[idx], y_pred_binary))
        accs.append(balanced_accuracy_score(y_true[idx], y_pred_binary))
        tn, fp, fn, tp = confusion_matrix(y_true[idx], y_pred_binary).ravel()
        
        sens.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        spes.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        pres.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
    
    # Print results
    print(f'sensitivity: {np.mean(sens):.3f} 95%CI ({np.percentile(sens, 2.5):.3f} - {np.percentile(sens, 97.5):.3f})')
    print(f'specificity: {np.mean(spes):.3f} 95%CI ({np.percentile(spes, 2.5):.3f} - {np.percentile(spes, 97.5):.3f})')
    print(f'precision: {np.mean(pres):.3f} 95%CI ({np.percentile(pres, 2.5):.3f} - {np.percentile(pres, 97.5):.3f})')
    print(f'f1 score: {np.mean(f1s):.3f} 95%CI ({np.percentile(f1s, 2.5):.3f} - {np.percentile(f1s, 97.5):.3f})')
    print(f'balanced accuracy: {np.mean(accs):.3f} 95%CI ({np.percentile(accs, 2.5):.3f} - {np.percentile(accs, 97.5):.3f})')
    
    # Plot AUROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    lower_auc = np.percentile(aucs, 2.5)
    upper_auc = np.percentile(aucs, 97.5)
    
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='b', alpha=.5)
    plt.plot(mean_fpr, mean_tpr, color='green',
             label=f'AUROC: {mean_auc:.3f} ({lower_auc:.3f} - {upper_auc:.3f})',
             lw=2, alpha=.8)
    
    lower_tprs = np.percentile(tprs, 2.5, axis=0)
    upper_tprs = np.percentile(tprs, 97.5, axis=0)
    plt.fill_between(mean_fpr, lower_tprs, upper_tprs, color='green', alpha=0.2)
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUROC curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'{plotdir}/auroc_ci.png', dpi=300, bbox_inches="tight")
    plt.close()
    
    # Plot AUPRC
    mean_precision = np.mean(precision_arrs, axis=0)
    lower_precision = np.percentile(precision_arrs, 2.5, axis=0)
    upper_precision = np.percentile(precision_arrs, 97.5, axis=0)
    mean_auprc = np.mean(auprcs)
    lower_auprc = np.percentile(auprcs, 2.5)
    upper_auprc = np.percentile(auprcs, 97.5)
    
    plt.figure(figsize=(5, 5))
    plt.plot(mean_recall, mean_precision, color='orange',
             label=f'AUPRC: {mean_auprc:.3f} ({lower_auprc:.3f} - {upper_auprc:.3f})',
             lw=2, alpha=.8)
    plt.fill_between(mean_recall, lower_precision, upper_precision, 
                     color='orange', alpha=0.2)
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-recall curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'{plotdir}/auprc_ci.png', dpi=300, bbox_inches="tight")
    plt.close()
    
    return {
        'auroc': mean_auc, 'auroc_lower': lower_auc, 'auroc_upper': upper_auc,
        'auprc': mean_auprc, 'auprc_lower': lower_auprc, 'auprc_upper': upper_auprc,
        'sensitivity': np.mean(sens), 'specificity': np.mean(spes),
        'precision': np.mean(pres), 'f1': np.mean(f1s), 'accuracy': np.mean(accs)
    }


def time_plot_auc_prc_ci(result_df, thval, time_hours=[24, 18, 12, 6, 3, 1], 
                          num_bootstraps=2000, plotdir='./plots'):
    """
    Time-stratified AUROC and AUPRC with bootstrap CI.
    
    "The secondary outcomes included the occurrences of cardiac arrest 
     from 0.5 to 18, 12, 6, 3, and 1 h"
    
    Args:
        result_df: DataFrame with 'time', 'y_true', 'y_prob' columns
        thval: Classification threshold
        time_hours: Prediction horizons to evaluate (in hours)
        num_bootstraps: Number of bootstrap iterations
        plotdir: Directory for output plots
    
    Returns:
        dict: Results for each time window
    
    Output Files:
        - auroc_by_time.png: Time-stratified ROC curves
        - auprc_by_time.png: Time-stratified PR curves
    """
    os.makedirs(plotdir, exist_ok=True)
    
    time_minutes = np.array(time_hours) * 60
    results = {}
    
    for t in time_minutes:
        # Filter: events within time window 
        data = result_df[(result_df['time'] <= t) | (result_df['time'] >= CRITERIA)]
        print(f'\nTime window: {t//60}h | N samples: {len(data)}')
        
        y_true = data['y_true'].values
        y_prob = data['y_prob'].values
        
        if len(np.unique(y_true)) < 2:
            print(f"  Skipping - only one class present")
            continue
        
        # Bootstrap arrays
        tprs, aucs = [], []
        auprcs, precision_arrs = [], []
        sens, spes, pres, f1s, accs = [], [], [], [], []
        mean_fpr = np.linspace(0, 1, 100)
        mean_recall = np.linspace(0, 1, 100)
        
        for _ in range(num_bootstraps):
            idx = np.random.choice(range(len(y_true)), size=len(y_true), replace=True)
            
            if len(np.unique(y_true[idx])) < 2:
                continue
            
            # ROC
            fpr, tpr, _ = roc_curve(y_true[idx], y_prob[idx])
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            aucs.append(auc(fpr, tpr))
            
            # PR
            precision, recall, _ = precision_recall_curve(y_true[idx], y_prob[idx])
            precision_arr = np.interp(mean_recall, recall[::-1], precision[::-1])
            auprcs.append(auc(mean_recall, precision_arr))
            precision_arrs.append(precision_arr)
            
            # Threshold-based metrics
            y_pred = y_prob[idx] > thval
            f1s.append(f1_score(y_true[idx], y_pred))
            accs.append(balanced_accuracy_score(y_true[idx], y_pred))
            tn, fp, fn, tp = confusion_matrix(y_true[idx], y_pred).ravel()
            sens.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
            spes.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
            pres.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
        
        print(f'  AUROC: {np.mean(aucs):.3f} ({np.percentile(aucs, 2.5):.3f} - {np.percentile(aucs, 97.5):.3f})')
        print(f'  AUPRC: {np.mean(auprcs):.3f} ({np.percentile(auprcs, 2.5):.3f} - {np.percentile(auprcs, 97.5):.3f})')
        print(f'  Sensitivity: {np.mean(sens):.3f} ({np.percentile(sens, 2.5):.3f} - {np.percentile(sens, 97.5):.3f})')
        print(f'  Specificity: {np.mean(spes):.3f} ({np.percentile(spes, 2.5):.3f} - {np.percentile(spes, 97.5):.3f})')
        print(f'  F1: {np.mean(f1s):.3f} ({np.percentile(f1s, 2.5):.3f} - {np.percentile(f1s, 97.5):.3f})')
        
        results[t] = {
            'aucs': aucs, 'auprcs': auprcs,
            'tprs': tprs, 'precision_arrs': precision_arrs,
            'sens': sens, 'spes': spes, 'pres': pres, 'f1s': f1s, 'accs': accs
        }
    
    # Plot time-stratified AUROC
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='gray', alpha=.5)
    
    for t in time_minutes:
        if t not in results:
            continue
        aucs = results[t]['aucs']
        tprs = results[t]['tprs']
        
        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)
        lower_auc = np.percentile(aucs, 2.5)
        upper_auc = np.percentile(aucs, 97.5)
        
        plt.plot(mean_fpr, mean_tpr,
                 label=f'{int(t/60)}h: {mean_auc:.3f} ({lower_auc:.3f} - {upper_auc:.3f})',
                 lw=2, alpha=.8)
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve by prediction horizon')
    plt.legend(loc='lower right', prop={'size': 10})
    plt.tight_layout()
    plt.savefig(f'{plotdir}/auroc_by_time.png', dpi=300, bbox_inches="tight")
    plt.close()
    
    # Plot time-stratified AUPRC
    plt.figure(figsize=(6, 6))
    
    for t in time_minutes:
        if t not in results:
            continue
        auprcs = results[t]['auprcs']
        precision_arrs = results[t]['precision_arrs']
        
        mean_recall = np.linspace(0, 1, 100)
        mean_precision = np.mean(precision_arrs, axis=0)
        mean_auprc = np.mean(auprcs)
        lower_auprc = np.percentile(auprcs, 2.5)
        upper_auprc = np.percentile(auprcs, 97.5)
        
        plt.plot(mean_recall, mean_precision,
                 label=f'{int(t/60)}h: {mean_auprc:.3f} ({lower_auprc:.3f} - {upper_auprc:.3f})',
                 lw=2, alpha=.8)
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR curve by prediction horizon')
    plt.legend(loc='lower right', prop={'size': 10})
    plt.tight_layout()
    plt.savefig(f'{plotdir}/auprc_by_time.png', dpi=300, bbox_inches="tight")
    plt.close()
    
    return results


# ============================================================
# Calibration Functions
# ============================================================

def fine_cali(y_prob, y_true, threshold=1.0, n_bins=10, mode='uniform', 
              save_path=None):
    """
    Calibration plot with Clopper-Pearson confidence intervals.

    Args:
        y_prob: Predicted probabilities
        y_true: Ground truth labels
        threshold: Maximum probability to display
        n_bins: Number of calibration bins
        mode: 'uniform' (equal-width) or 'quantile' (equal-count) bins
        save_path: Path to save the plot
    
    Returns:
        pd.DataFrame: Calibration data per bin
    """
    # Clip probabilities
    y_prob = np.where(y_prob > threshold, threshold, y_prob)
    
    # Create bins
    bin_edges = np.linspace(0, threshold, n_bins + 1)
    data = pd.DataFrame({'y_true': y_true, 'y_pred_proba': y_prob})
    
    if mode == 'uniform':
        data['bin'] = pd.cut(data['y_pred_proba'], bins=bin_edges, 
                             include_lowest=True, right=False)
    elif mode == 'quantile':
        data['bin'] = pd.qcut(data['y_pred_proba'], q=n_bins, duplicates='drop')
    
    # Calculate calibration data
    calibration_data = data.groupby('bin').agg(
        y_pred_proba_mean=('y_pred_proba', 'mean'),
        y_true_mean=('y_true', 'mean'),
        y_true_count=('y_true', 'count'),
        y_true_sum=('y_true', 'sum')
    ).reset_index()
    
    # Clopper-Pearson CI
    alpha = 0.05
    calibration_data['y_true_lower'] = beta.ppf(
        alpha / 2, 
        calibration_data['y_true_sum'], 
        calibration_data['y_true_count'] - calibration_data['y_true_sum'] + 1
    )
    calibration_data['y_true_upper'] = beta.ppf(
        1 - alpha / 2, 
        calibration_data['y_true_sum'] + 1, 
        calibration_data['y_true_count'] - calibration_data['y_true_sum']
    )
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(calibration_data['y_pred_proba_mean'], calibration_data['y_true_mean'], 
             marker='o', linestyle='None', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated', 
             linewidth=1, color='gray')
    
    # Error bars
    for _, row in calibration_data.iterrows():
        plt.vlines(x=row['y_pred_proba_mean'], 
                   ymin=row['y_true_lower'], ymax=row['y_true_upper'],
                   linewidth=1)
        plt.hlines(y=row['y_true_lower'],
                   xmin=row['y_pred_proba_mean'] - 0.01,
                   xmax=row['y_pred_proba_mean'] + 0.01, linewidth=1)
        plt.hlines(y=row['y_true_upper'],
                   xmin=row['y_pred_proba_mean'] - 0.01,
                   xmax=row['y_pred_proba_mean'] + 0.01, linewidth=1)
    
    plt.xlim(0, threshold)
    plt.ylim(0, threshold)
    plt.xlabel('Expected probability of cardiac arrest')
    plt.ylabel('Observed fraction of cardiac arrest')
    plt.legend(loc='upper left')
    plt.grid(False)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    # Compute Brier score
    brier = brier_score_loss(y_true, y_prob)
    print(f'Brier score: {brier:.4f}')
    
    return calibration_data


def time_fine_cali(result_df, time_hours=[24, 18, 12, 6, 3, 1], 
                   threshold=1.0, n_bins=10, mode='uniform', save_path=None):
    """
    Time-stratified calibration plots.
    
    Args:
        result_df: DataFrame with 'time', 'y_true', 'y_prob' columns
        time_hours: Prediction horizons (in hours)
        threshold: Maximum probability to display
        n_bins: Number of calibration bins
        mode: Binning strategy
        save_path: Path to save the combined plot
    """
    n_cols = ceil(len(time_hours) / 2)
    n_rows = 2
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    time_minutes = np.array(time_hours) * 60
    
    for ax, t in zip(axs.flatten(), time_minutes):
        data = result_df[(result_df['time'] <= t) | (result_df['time'] >= CRITERIA)]
        print(f'Time: {t//60}h | N samples: {len(data)} | N positive: {sum(data["y_true"])}')
        
        y_prob = data['y_prob'].values
        y_true = data['y_true'].values
        
        # Clip and bin
        y_prob = np.where(y_prob > threshold, threshold, y_prob)
        bin_edges = np.linspace(0, threshold, n_bins + 1)
        df = pd.DataFrame({'y_true': y_true, 'y_pred_proba': y_prob})
        
        if mode == 'uniform':
            df['bin'] = pd.cut(df['y_pred_proba'], bins=bin_edges, 
                               include_lowest=True, right=False)
        else:
            df['bin'] = pd.qcut(df['y_pred_proba'], q=n_bins, duplicates='drop')
        
        calibration_data = df.groupby('bin').agg(
            y_pred_proba_mean=('y_pred_proba', 'mean'),
            y_true_mean=('y_true', 'mean'),
            y_true_count=('y_true', 'count'),
            y_true_sum=('y_true', 'sum')
        ).reset_index()
        
        alpha = 0.05
        calibration_data['y_true_lower'] = beta.ppf(
            alpha / 2, calibration_data['y_true_sum'],
            calibration_data['y_true_count'] - calibration_data['y_true_sum'] + 1)
        calibration_data['y_true_upper'] = beta.ppf(
            1 - alpha / 2, calibration_data['y_true_sum'] + 1,
            calibration_data['y_true_count'] - calibration_data['y_true_sum'])
        
        ax.plot(calibration_data['y_pred_proba_mean'], calibration_data['y_true_mean'],
                marker='o', linestyle='None', label='Model')
        ax.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated',
                linewidth=1, color='gray')
        
        for _, row in calibration_data.iterrows():
            ax.vlines(x=row['y_pred_proba_mean'],
                      ymin=row['y_true_lower'], ymax=row['y_true_upper'], linewidth=1)
            ax.hlines(y=row['y_true_lower'],
                      xmin=row['y_pred_proba_mean'] - 0.01,
                      xmax=row['y_pred_proba_mean'] + 0.01, linewidth=1)
            ax.hlines(y=row['y_true_upper'],
                      xmin=row['y_pred_proba_mean'] - 0.01,
                      xmax=row['y_pred_proba_mean'] + 0.01, linewidth=1)
        
        ax.set_xlim(0, threshold)
        ax.set_ylim(0, threshold)
        ax.set_xlabel('Expected probability')
        ax.set_ylabel('Observed fraction')
        ax.set_title(f'Time: {t//60}h')
        ax.legend(loc='upper left')
        ax.grid(False)
    
    # Remove unused subplots
    for i in range(len(time_hours), n_rows * n_cols):
        fig.delaxes(axs.flatten()[i])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# SHAP Functions
# ============================================================

def shap_plot(model, x_test, feature_names, save_path=None):
    """
    SHAP summary plot for feature importance.
    
    Args:
        model: Trained LightGBM Booster
        x_test: Test data
        feature_names: List of feature names
        save_path: Path to save the plot
    """
    try:
        import shap
        
        explainer = shap.explainers.Tree(model, model_output='raw')
        shap_values = explainer(x_test.to_numpy() if hasattr(x_test, 'to_numpy') else x_test)
        
        plt.figure()
        shap.summary_plot(shap_values, x_test, feature_names=feature_names, 
                          show=False, max_display=20)
        plt.title('Feature importances (SHAP)')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        
    except ImportError:
        print("SHAP not installed. Skipping SHAP plot.")
    except Exception as e:
        print(f"SHAP plot failed: {e}")
