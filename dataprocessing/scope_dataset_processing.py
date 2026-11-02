"""
SCOPE Dataset Processing for Cardiac Arrest Prediction
=======================================================

Implementation of ECG preprocessing and HRV feature extraction pipeline

This module processes raw ECG signals from the SCOPE dataset to extract
Heart Rate Variability (HRV) features for cardiac arrest prediction models.

Pipeline Overview:
    Step 1: ECG signal loading and resampling
    Step 2: Signal segmentation into 5-minute epochs
    Step 3: Noise filtering (flat signal detection + Butterworth high-pass + powerline removal)
    Step 4: R-peak detection via Neurokit2
    Step 5: Quality assessment using QRS segment distance from average
    Step 6: HRV feature extraction (43 initial feature candidates)

Note:
    ICU stay-level parallel processing is implemented to accelerate computation
    across large patient cohorts.    

Primary Outcome:
    Prediction of cardiac arrest occurrence within 0.5–24 hours
"""

import argparse
import os
import random
import pandas as pd
import numpy as np
import neurokit2 as nk
from multiprocessing import Pool, cpu_count
from neurokit2.signal import signal_filter
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# Configuration Parameters
# ============================================================

# --- Processing Options ---
USE_PARALLEL = True     # Enable parallel processing
N_WORKERS = 8           # Number of parallel workers
MAX_CASES = None        # Maximum cases per group (None = all)
SPECIFIC_IDS = None     # Process only specific case IDs (None = all, e.g., [1, 2, 3])


# --- Signal Acquisition Parameters ---
# ECG signals processed at 250 Hz to reduce computational requirements
# Note: Resampling is handled by VitalDB's to_numpy(interval=) parameter,
#       so source data at any sampling rate will be converted to TARGET_SRATE
TARGET_SRATE = 250      # Target sampling rate (Hz)
INTERVAL = 1 / TARGET_SRATE  # Sampling interval for VitalDB extraction

# --- Time Window Parameters ---
# Primary outcome: cardiac arrest within 0.5–24 hours
HORIZON_MAX_HOUR = 24   # Prediction horizon upper bound: up to 24h before cardiac arrest
HORIZON_MIN_MIN = 30    # Prediction horizon lower bound: at least 0.5h (30min) before cardiac arrest
WINDOW_MIN = 5          # ECG epoch duration: 5 minutes
SLIDE_MIN = 5           # Sliding window interval: 5 minutes

# --- Quality Control Parameters ---
# Quality assessment:
# "When the proportion of QRS segments with a quality index >= 0.9 
#  was not greater than 80%, epochs were excluded"
CONSEC_SEC = 10         # Maximum consecutive flat/NaN duration allowed (seconds)
QUALITY_THRESHOLD = 0.9  # QRS quality index threshold (1=closest to avg, 0=most distant)
QUALITY_RATIO = 0.8     # Minimum proportion of high-quality QRS segments required

SEED = 42               # Random seed for reproducibility

# --- HRV Feature Set ---
# 43 HRV features as initial feature candidates before BorutaShap selection
# Derived from Neurokit2 toolkit comprising:
#   - Time-domain measures (20 features): MeanNN, SDNN, RMSSD, etc.
#   - Frequency-domain measures (9 features): ULF, VLF, LF, HF, VHF, ratios
#   - Nonlinear measures (14 features): Poincaré plot (SD1, SD2) and heart rate fragmentation indices
HRV_PARAMS = [
    # === Time-Domain Measures ===
    # Basic statistical measures of RR intervals
    'HRV_MeanNN',       # Mean of RR intervals
    'HRV_SDNN',         # Standard deviation of RR intervals
    'HRV_SDANN1',       # SD of average RR intervals from 1-min segments
    'HRV_SDNNI1',       # Mean of SD of RR intervals from 1-min segments
    'HRV_RMSSD',        # Root mean square of successive RR differences
    'HRV_SDSD',         # SD of successive differences between RR intervals
    
    # Coefficient of variation measures
    'HRV_CVNN',         # SDNN divided by MeanNN
    'HRV_CVSD',         # RMSSD divided by MeanNN
    
    # Robust statistical measures
    'HRV_MedianNN',     # Median of RR intervals
    'HRV_MadNN',        # Median absolute deviation of RR intervals
    'HRV_MCVNN',        # MadNN divided by MedianNN
    'HRV_IQRNN',        # Interquartile range of RR intervals
    'HRV_Prc20NN',      # 20th percentile of RR intervals
    'HRV_Prc80NN',      # 80th percentile of RR intervals
    
    # pNNxx measures
    'HRV_pNN50',        # Percentage of absolute successive RR differences > 50ms
    'HRV_pNN20',        # Percentage of absolute successive RR differences > 20ms
    
    # Range measures
    'HRV_MinNN',        # Minimum of RR intervals
    'HRV_MaxNN',        # Maximum of RR intervals
    
    # Geometric measures
    'HRV_HTI',          # Total RR intervals divided by histogram height
    'HRV_TINN',         # Baseline width of RR distribution by triangular interpolation
    
    # === Frequency-Domain Measures ===
    # Spectral power in different frequency bands
    'HRV_ULF',          # Ultra-low frequency power
    'HRV_VLF',          # Very low frequency power
    'HRV_LF',           # Low frequency power
    'HRV_HF',           # High frequency power
    'HRV_VHF',          # Very high frequency power
    
    # Frequency ratios and normalized measures
    'HRV_LFHF',         # LF/HF ratio
    'HRV_LFn',          # Normalized LF (LF divided by total power)
    'HRV_HFn',          # Normalized HF (HF divided by total power)
    'HRV_LnHF',         # Log transformed HF power
    
    # === Nonlinear Measures ===
    # Poincaré plot measures
    'HRV_SD1',          # SD perpendicular to line of identity
    'HRV_SD2',          # SD along line of identity
    'HRV_SD1SD2',       # Ratio of SD1 to SD2
    'HRV_S',            # Area of ellipse described by SD1 and SD2
    'HRV_CSI',          # Cardiac Sympathetic Index
    'HRV_CVI',          # Cardiac Vagal Index
    'HRV_CSI_Modified', # Modified CSI
    
    # Heart Rate Fragmentation measures
    'HRV_PIP',          # Percentage of inflection points of the RR intervals series
    'HRV_IALS',         # Inverse average length of acceleration/deceleration segments
    'HRV_PSS',          # Percentage of short segments
    'HRV_PAS',          # Percentage of alternation segments
    
    # Asymmetry measures
    'HRV_GI',           # Guzik's index
    'HRV_SI',           # Slope index
    'HRV_AI',           # Area index
    'HRV_PI',           # Porta's index
]


# ============================================================
# Worker Function for Multiprocessing
# ============================================================
def _process_case_worker(args):
    """
    Worker function for parallel stay-level ECG processing.
    
    This function implements the complete preprocessing pipeline for a single
    ICU stay, including ECG loading, segmentation, quality control,
    and HRV feature extraction.
    
    Must be defined at module level for pickle serialization in multiprocessing.
    
    Args:
        args (tuple): Contains:
            - case_id: Stay identifier
            - label: Outcome label (1=cardiac arrest, 0=discharge)
            - data_path: Path to SCOPE dataset
            - srate: Target sampling rate (Hz)
            - horizon_max_hour: Prediction horizon upper bound (hours)
            - window_min: Epoch duration (minutes)
            - slide_min: Sliding window step (minutes)
            - horizon_min_min: Prediction horizon lower bound (minutes)
            - seed: Random seed for reproducibility
    
    Returns:
        tuple: (list of processed segment dictionaries, case statistics dict)
    """
    case_id, label, data_path, srate, horizon_max_hour, window_min, slide_min, horizon_min_min, seed = args
    
    import vitaldb
    
    # --- Initialize Case Statistics Tracking ---
    case_stats = {
        'case_id': case_id,
        'label': label,
        'ecg_loaded': False,
        'ecg_duration_hours': 0,
        'total_segments': 0,
        'excluded_flat': 0,          # Excluded due to flat/NaN segments
        'excluded_quality': 0,       # Excluded due to low QRS quality ratio
        'excluded_hrv_error': 0,     # Excluded due to HRV computation failure
        'processed_segments': 0,
        'avg_quality_ratio': np.nan,
        'error_message': None
    }
    
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # =========================================================
    # Step 1: ECG Signal Loading
    # =========================================================
    vital_path = os.path.join(data_path, f"{case_id}.vital")
    if not os.path.exists(vital_path):
        case_stats['error_message'] = 'Vital file not found'
        return [], case_stats
    
    try:
        vf = vitaldb.VitalFile(vital_path)
        
        # --- Time Window Selection Based on Outcome ---
        # For cardiac arrest cases: Extract window ending at arrest time
        # For control (discharge) cases: Random window from entire stay
        if label == 1:  # Cardiac arrest case
            # Crop to horizon_max_hour hours prior to cardiac arrest (outcome timestamp)
            outcome_ts = vf.get_dt(2100, 1, 1, 0, 0, 0)
            start_ts = outcome_ts - (horizon_max_hour * 60 * 60)
            vf.crop(start_ts, outcome_ts)
        # Control cases: No cropping - random sampling applied later
        
        # --- ECG Channel Selection ---
        # Primary: ECG Lead II (standard for HRV analysis)
        # Fallback: Any available ECG channel (excluding heart rate trends)
        ecg_data = vf.to_numpy('ECG_II', interval=INTERVAL)
        
        if ecg_data is None or len(ecg_data) == 0:
            tracks = vf.get_track_names()
            ecg_tracks = [t for t in tracks if 'ECG' in t and 'HR' not in t]
            for track in ecg_tracks:
                ecg_data = vf.to_numpy(track, interval=INTERVAL)
                if ecg_data is not None and len(ecg_data) > 0:
                    break
        
        if ecg_data is None or len(ecg_data) == 0:
            case_stats['error_message'] = 'No ECG data found'
            return [], case_stats
        
        case_stats['ecg_loaded'] = True
        case_stats['ecg_duration_hours'] = len(ecg_data) / (srate * 3600)
            
    except Exception as e:
        case_stats['error_message'] = f'ECG load error: {str(e)}'
        return [], case_stats
    
    # =========================================================
    # Step 2: ECG Signal Segmentation into 5-minute Epochs
    # =========================================================
    # "ECG signals were divided into 5 min epochs"
    window_samples = int(window_min * 60 * srate)   # 5 min = 75,000 samples at 250 Hz
    slide_samples = int(slide_min * 60 * srate)     # 5 min sliding step
    total_samples = len(ecg_data)
    
    segments = []
    
    if label == 1:  # Cardiac Arrest Case (EVENT group)
        # --- Prediction Horizon Constraints ---
        # Primary outcome: cardiac arrest within 0.5–24 hours
        # Exclude segments outside the prediction horizon
        min_samples_to_event = int(horizon_min_min * 60 * srate)           # 30 min (0.5h) lower bound
        max_samples_to_event = int(horizon_max_hour * 60 * 60 * srate)     # 24 hours upper bound
        
        for start in range(0, total_samples - window_samples + 1, slide_samples):
            end = start + window_samples
            segment = ecg_data[start:end]
            
            # Calculate time remaining until cardiac arrest
            samples_from_end = total_samples - end
            time_to_event_min = samples_from_end / (srate * 60)
            
            # Apply prediction horizon filter (0.5h to 24h before event)
            if samples_from_end < min_samples_to_event or samples_from_end > max_samples_to_event:
                continue
            
            segments.append({
                'segment': segment,
                'label': 1,
                'case_id': case_id,
                'time_to_event': time_to_event_min
            })
    
    else:  # Discharge Case (CONTROL group)
        # --- Random Window Selection for Controls ---
        # For patients with >24H data, randomly select a matching window
        # This ensures comparable observation periods between groups
        horizon_max_samples = int(horizon_max_hour * 60 * 60 * srate)
        
        if total_samples <= horizon_max_samples:
            start_idx = 0
            end_idx = total_samples
        else:
            max_start = total_samples - horizon_max_samples
            start_idx = random.randint(0, max_start)
            end_idx = start_idx + horizon_max_samples
        
        ecg_window = ecg_data[start_idx:end_idx]
        
        for start in range(0, len(ecg_window) - window_samples + 1, slide_samples):
            end = start + window_samples
            segment = ecg_window[start:end]

            # Calculate time from segment end to window end
            # (for consistency with case group time_to_event metric)
            samples_from_end = len(ecg_window) - end
            time_to_event_min = samples_from_end / (srate * 60)            
            
            segments.append({
                'segment': segment,
                'label': 0,
                'case_id': case_id,
                'time_to_event': time_to_event_min
            })
    
    case_stats['total_segments'] = len(segments)
    
    if len(segments) == 0:
        case_stats['error_message'] = 'No valid segments in time window'
        return [], case_stats
    
    # =========================================================
    # Step 3-6: Signal Processing and HRV Extraction per Epoch
    # =========================================================
    processed = []
    quality_ratios = []
    
    for seg_info in segments:
        segment = seg_info['segment'].squeeze()
        
        # ---------------------------------------------------------
        # Step 3a: Flat Signal Detection
        # ---------------------------------------------------------
        # Exclude epochs with extended flat regions (consecutive identical values)
        # or prolonged NaN sequences, indicating signal dropout or artifact
        flat_threshold = srate * CONSEC_SEC  # 10 seconds threshold
        
        # Check for consecutive NaN values
        nan_mask = np.isnan(segment)
        if nan_mask.any():
            nan_segments = np.split(nan_mask, np.where(np.diff(nan_mask))[0] + 1)
            nan_long = [s for s in nan_segments if s[0] == True and len(s) > flat_threshold]
            if len(nan_long) > 0:
                case_stats['excluded_flat'] += 1
                continue
        
        # Check for consecutive identical values (flat line)
        con_list = np.split(segment, np.where(np.diff(segment) != 0)[0] + 1)
        con_count = [d for d in con_list if len(d) > flat_threshold]
        if len(con_count) > 0:
            case_stats['excluded_flat'] += 1
            continue
        
        try:
            # ---------------------------------------------------------
            # Step 3b: ECG Signal Filtering
            # ---------------------------------------------------------
            # "A 0.5 Hz high-pass Butterworth filter (order=5) was used,
            #  followed by powerline filtering of the 5 min ECG signal"
            
            # High-pass filter: Remove baseline wander (>0.5 Hz)
            cleaned = signal_filter(
                signal=segment, 
                sampling_rate=srate, 
                lowcut=0.5,              # 0.5 Hz high-pass cutoff
                method="butterworth", 
                order=5             
            )
            
            # Powerline filter: Remove 60 Hz electrical interference
            cleaned = signal_filter(
                signal=cleaned, 
                sampling_rate=srate, 
                method="powerline", 
                powerline=60              # 60 Hz powerline frequency
            )
            
            # ---------------------------------------------------------
            # Step 4: R-Peak Detection
            # ---------------------------------------------------------
            # "Neurokit2 Python library, a comprehensive and validated toolkit
            #  for ECG signal analysis, was utilized to detect R peaks"
            _, peaks = nk.ecg_peaks(
                cleaned, 
                sampling_rate=srate, 
                method='neurokit',        # Default Neurokit2 detection
                correct_artifacts=True     # Artifact correction enabled
            )
            rpeaks = peaks['ECG_R_Peaks']
            
            # ---------------------------------------------------------
            # Step 5: Quality Assessment
            # ---------------------------------------------------------
            # "A continuous quality index was computed by interpolating the
            #  distance of each QRS segment from the average QRS segment
            #  (1 = closest to average, 0 = most distant).
            #  When the proportion of QRS segments with quality index >= 0.9
            #  was not greater than 80%, epochs were excluded."
            quality_index = nk.ecg_quality(
                cleaned, 
                rpeaks=rpeaks, 
                sampling_rate=srate, 
                method='averageQRS'       # QRS morphology consistency-based ECG quality
            )
            quality_ratio = (quality_index >= QUALITY_THRESHOLD).sum() / len(quality_index)
            
            if quality_ratio < QUALITY_RATIO:
                case_stats['excluded_quality'] += 1
                continue
            
            # ---------------------------------------------------------
            # Step 6: HRV Feature Extraction
            # ---------------------------------------------------------
            # "The toolkit facilitated the calculation of HRV measures based
            #  on detected R peaks, ensuring reliable results through a 
            #  standardized and automated approach."
            hrv_df = nk.hrv(rpeaks, sampling_rate=srate, show=False)
            
            if hrv_df.empty:
                case_stats['excluded_hrv_error'] += 1
                continue
            
            # Extract the 43 pre-selected HRV features
            hrv_dict = {}
            for param in HRV_PARAMS:
                if param in hrv_df.columns:
                    hrv_dict[param] = hrv_df[param].iloc[0]
                else:
                    hrv_dict[param] = np.nan
            
            # Compile segment results
            result = {
                'case_id': seg_info['case_id'],
                'label': seg_info['label'],
                'time_to_event': seg_info['time_to_event'],
                'quality_ratio': quality_ratio,
                **hrv_dict
            }
            processed.append(result)
            quality_ratios.append(quality_ratio)
            
        except Exception as e:
            case_stats['excluded_hrv_error'] += 1
            continue
    
    # Update final statistics
    case_stats['processed_segments'] = len(processed)
    if quality_ratios:
        case_stats['avg_quality_ratio'] = np.mean(quality_ratios)
    
    return processed, case_stats


# ============================================================
# Main Dataset Class
# ============================================================
class SCOPEDataset:
    """
    Dataset class for SCOPE cardiac arrest prediction.
    
    The class handles:
        - Metadata loading and patient group assignment (case vs. control)
        - Parallel processing of ECG signals across multiple patients
        - HRV feature extraction with quality control
        - Result saving and processing statistics reporting
    
    Attributes:
        data_path (str): Path to SCOPE dataset directory
        horizon_max_hour (int): Prediction horizon upper bound (hours)
        window_min (int): ECG epoch duration (minutes)
        slide_min (int): Sliding window step (minutes)
        horizon_min_min (int): Prediction horizon lower bound (minutes)
        srate (int): Target sampling rate (Hz)
        seed (int): Random seed for reproducibility
        metadata (pd.DataFrame): Patient metadata
        case_df (pd.DataFrame): Cardiac arrest cases
        control_df (pd.DataFrame): Discharge (control) cases
    
    Example:
        >>> dataset = SCOPEDataset(data_path='input/SCOPE')
        >>> df, stats = dataset.load_all_data(use_parallel=True, n_workers=8)
        >>> X, y, case_ids = dataset.get_feature_matrix(df)
    """

    def __init__(
        self,
        data_path: str,
        horizon_max_hour: int = HORIZON_MAX_HOUR,
        window_min: int = WINDOW_MIN,
        slide_min: int = SLIDE_MIN,
        horizon_min_min: int = HORIZON_MIN_MIN,
        srate: int = TARGET_SRATE,
        seed: int = SEED
    ):
        """
        Initialize SCOPE dataset processor.
        
        Args:
            data_path: Path to SCOPE dataset containing .vital files
            horizon_max_hour: Prediction horizon upper bound (default: 24h)
            window_min: ECG epoch duration (default: 5min)
            slide_min: Sliding window step (default: 5min)
            horizon_min_min: Prediction horizon lower bound (default: 30min = 0.5h)
            srate: Target sampling rate after downsampling (default: 250 Hz)
            seed: Random seed for reproducibility
        """
        self.data_path = data_path
        self.horizon_max_hour = horizon_max_hour
        self.window_min = window_min
        self.slide_min = slide_min
        self.horizon_min_min = horizon_min_min
        self.srate = srate
        self.seed = seed

        self.metadata = None
        self.case_df = None
        self.control_df = None

        self._load_metadata()

    def _load_metadata(self):
        """
        Load and parse SCOPE metadata file.
        
        Separates patients into groups based on outcome:
            - Case (Cardiac Arrest): Primary prediction target
            - Control (Discharge): Negative class
            - Death (non-cardiac): Excluded from analysis
        """
        meta_path = os.path.join(self.data_path, 'metadata_v1.0.xlsx')
        self.metadata = pd.read_excel(meta_path)

        # Group assignment based on outcome
        self.case_df = self.metadata[self.metadata['outcome'] == 'Cardiac Arrest'].copy()
        self.control_df = self.metadata[self.metadata['outcome'] == 'Discharge'].copy()
        self.death_df = self.metadata[self.metadata['outcome'] == 'Death'].copy()

        print(f"[INFO] Metadata loaded:")
        print(f"  - Case (Cardiac arrest): {len(self.case_df)}")
        print(f"  - Control (Discharge): {len(self.control_df)}")
        print(f"  - Excluded (Death): {len(self.death_df)}")

    def load_all_data(self, max_cases: int = None, use_parallel: bool = True, 
                  n_workers: int = None, specific_ids: list = None) -> tuple:
        """
        Load and process all cases and controls.
        
        Implements stay-level parallel processing for efficient computation
        across large patient cohorts.
        
        Args:
            max_cases: Maximum patients per group (None = all)
            use_parallel: Enable multiprocessing (default: True)
            n_workers: Number of parallel workers (None = CPU count)
            specific_ids: Process only specified case IDs (for debugging)
            
        Returns:
            tuple: (
                DataFrame with HRV features for all valid segments,
                DataFrame with per-case processing statistics
            )
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        if n_workers is None:
            n_workers = cpu_count()

        all_results = []
        all_stats = []

        # Filter to specific IDs if provided (for debugging/testing)
        if specific_ids is not None:
            case_df = self.case_df[self.case_df['case_id'].isin(specific_ids)]
            control_df = self.control_df[self.control_df['case_id'].isin(specific_ids)]
        else:     
            case_df = self.case_df
            control_df = self.control_df

        # Sample subset if max_cases specified
        if max_cases is not None:
            case_df = case_df.sample(n=min(max_cases, len(case_df)), random_state=self.seed)
            control_df = control_df.sample(n=min(max_cases, len(control_df)), random_state=self.seed)

        # Prepare worker arguments for parallel processing
        case_args = [
            (row['case_id'], 1, self.data_path, self.srate, self.horizon_max_hour, 
             self.window_min, self.slide_min, self.horizon_min_min, self.seed + idx)
            for idx, (_, row) in enumerate(case_df.iterrows())
        ]
        
        control_args = [
            (row['case_id'], 0, self.data_path, self.srate, self.horizon_max_hour,
             self.window_min, self.slide_min, self.horizon_min_min, self.seed + idx)
            for idx, (_, row) in enumerate(control_df.iterrows())
        ]

        if use_parallel:
            print(f"\n[INFO] Processing {len(case_df)} cases + {len(control_df)} controls with {n_workers} workers...")
            
            # Process cardiac arrest cases (EVENT group)
            print(f"\n[INFO] Processing cardiac arrest cases (EVENT group)...")
            with Pool(processes=n_workers) as pool:
                case_results = list(tqdm(
                    pool.imap(_process_case_worker, case_args),
                    total=len(case_args),
                    desc="Cases"
                ))
            for segments, stats in case_results:
                if segments:
                    all_results.extend(segments)
                all_stats.append(stats)
            
            # Process discharge cases (CONTROL group)
            print(f"\n[INFO] Processing control cases (CONTROL group)...")
            with Pool(processes=n_workers) as pool:
                control_results = list(tqdm(
                    pool.imap(_process_case_worker, control_args),
                    total=len(control_args),
                    desc="Controls"
                ))
            for segments, stats in control_results:
                if segments:
                    all_results.extend(segments)
                all_stats.append(stats)
        else:
            # Sequential processing
            print(f"\n[INFO] Processing {len(case_df)} cardiac arrest cases (EVENT group)...")
            for args in tqdm(case_args, desc="Cases"):
                segments, stats = _process_case_worker(args)
                if segments:
                    all_results.extend(segments)
                all_stats.append(stats)
            
            print(f"\n[INFO] Processing {len(control_df)} control cases (CONTROL group)...")
            for args in tqdm(control_args, desc="Controls"):
                segments, stats = _process_case_worker(args)
                if segments:
                    all_results.extend(segments)
                all_stats.append(stats)

        # Compile results into DataFrames
        df = pd.DataFrame(all_results)
        stats_df = pd.DataFrame(all_stats)

        # Handle missing values with median imputation
        if len(df) > 0:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        # Print processing summary
        print(f"\n[INFO] Processing complete:")
        print(f"  - Total segments: {len(df)}")
        if len(df) > 0:
            print(f"  - Case segments (label=1): {(df['label'] == 1).sum()}")
            print(f"  - Control segments (label=0): {(df['label'] == 0).sum()}")
            print(f"  - Unique cases: {df['case_id'].nunique()}")

        return df, stats_df

    def get_feature_matrix(self, df: pd.DataFrame) -> tuple:
        """
        Extract feature matrix and labels for model training/evaluation.
        
        Extracts the 43 HRV features as initial feature candidates
        for subsequent feature selection (e.g., BorutaShap algorithm).
        
        Args:
            df: DataFrame from load_all_data()
            
        Returns:
            tuple: (
                X: Feature matrix (n_segments × 43 HRV features),
                y: Binary labels (1=cardiac arrest, 0=discharge),
                case_ids: Group labels for cross-validation
            )
        """
        feature_cols = [col for col in df.columns if col.startswith('HRV_')]

        X = df[feature_cols].values
        y = df['label'].values
        case_ids = df['case_id'].values

        print(f"\n[INFO] Feature matrix prepared:")
        print(f"  - Shape: {X.shape}")
        print(f"  - Features: {len(feature_cols)}")
        print(f"  - Positive samples (label=1): {(y == 1).sum()}")
        print(f"  - Negative samples (label=0): {(y == 0).sum()}")

        return X, y, case_ids
    
    def save_results(self, df: pd.DataFrame, stats_df: pd.DataFrame, 
                     output_dir: str) -> dict:
        """
        Save processed results and statistics.
        
        Outputs:
            1. HRV features (Parquet format) - Main feature table
            2. Case statistics (Parquet format) - Per-stay processing stats
            3. Processing summary (TXT) - Human-readable report
        
        Args:
            df: HRV features DataFrame
            stats_df: Case statistics DataFrame
            output_dir: Output directory path
            
        Returns:
            dict: Paths to saved files
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_subdir = os.path.join(output_dir, timestamp)
        os.makedirs(output_subdir, exist_ok=True)
        
        # 1. Save HRV features (parquet for efficient storage)
        hrv_path = os.path.join(output_subdir, f'scope_hrv_features.parquet')
        df.to_parquet(hrv_path, index=False)
        print(f"[SAVED] HRV features: {hrv_path}")
        
        # 2. Save case statistics
        stats_path = os.path.join(output_subdir, f'scope_case_stats.parquet')
        stats_df.to_parquet(stats_path, index=False)
        print(f"[SAVED] Case statistics: {stats_path}")
        
        # 3. Generate human-readable processing summary
        summary_path = os.path.join(output_subdir, f'scope_processing_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("SCOPE Dataset Processing Summary\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            
            # Configuration section
            f.write("[Configuration]\n")
            f.write(f"  - Data path: {self.data_path}\n")
            f.write(f"  - Sampling rate: {self.srate} Hz\n")
            f.write(f"  - Prediction horizon: {self.horizon_min_min} min - {self.horizon_max_hour}h\n")
            f.write(f"  - Epoch duration: {self.window_min} min\n")
            f.write(f"  - Sliding step: {self.slide_min} min\n")
            f.write(f"  - Quality threshold: {QUALITY_THRESHOLD} (QRS distance index)\n")
            f.write(f"  - Quality ratio: {QUALITY_RATIO} (min proportion high-quality)\n")
            f.write(f"  - Flat signal threshold: {CONSEC_SEC}s\n")
            f.write(f"  - Random seed: {self.seed}\n\n")
            
            # Overall statistics
            f.write("[Overall Statistics]\n")
            f.write(f"  - Total cases processed: {len(stats_df)}\n")
            f.write(f"  - Cases (Cardiac Arrest): {(stats_df['label'] == 1).sum()}\n")
            f.write(f"  - Controls (Discharge): {(stats_df['label'] == 0).sum()}\n")
            f.write(f"  - Total segments: {len(df)}\n")
            if len(df) > 0:
                f.write(f"  - Case segments: {(df['label'] == 1).sum()}\n")
                f.write(f"  - Control segments: {(df['label'] == 0).sum()}\n\n")
            
            # Segment exclusion statistics
            f.write("[Segment Exclusion Statistics]\n")
            f.write(f"  - Total segments attempted: {stats_df['total_segments'].sum()}\n")
            f.write(f"  - Excluded (flat/NaN signal): {stats_df['excluded_flat'].sum()}\n")
            f.write(f"  - Excluded (QRS quality <{QUALITY_RATIO*100:.0f}%): {stats_df['excluded_quality'].sum()}\n")
            f.write(f"  - Excluded (HRV computation error): {stats_df['excluded_hrv_error'].sum()}\n")
            f.write(f"  - Final processed: {stats_df['processed_segments'].sum()}\n")
            
            total_attempted = stats_df['total_segments'].sum()
            if total_attempted > 0:
                success_rate = stats_df['processed_segments'].sum() / total_attempted * 100
                f.write(f"  - Success rate: {success_rate:.2f}%\n\n")
            
            # Per-group statistics
            f.write("[Per-Group Statistics]\n")
            for label, group_name in [(1, 'Case (Cardiac Arrest)'), (0, 'Control (Discharge)')]:
                group_stats = stats_df[stats_df['label'] == label]
                if len(group_stats) > 0:
                    f.write(f"\n  {group_name}:\n")
                    f.write(f"    - Cases: {len(group_stats)}\n")
                    f.write(f"    - Avg ECG duration: {group_stats['ecg_duration_hours'].mean():.2f}h\n")
                    f.write(f"    - Avg segments per case: {group_stats['total_segments'].mean():.1f}\n")
                    f.write(f"    - Avg processed per case: {group_stats['processed_segments'].mean():.1f}\n")
                    f.write(f"    - Avg quality ratio: {group_stats['avg_quality_ratio'].mean():.4f}\n")
            
            # Error cases
            error_cases = stats_df[stats_df['error_message'].notna()]
            if len(error_cases) > 0:
                f.write(f"\n[Cases with Errors] ({len(error_cases)} cases)\n")
                for _, row in error_cases.iterrows():
                    f.write(f"  - Case {row['case_id']}: {row['error_message']}\n")
            
            # HRV feature statistics
            f.write("\n[HRV Feature Statistics]\n")
            f.write(f"  - Total features: {len(HRV_PARAMS)} (initial candidates before BorutaShap)\n")
            hrv_cols = [col for col in df.columns if col.startswith('HRV_')]
            nan_counts = df[hrv_cols].isna().sum()
            nan_with_values = nan_counts[nan_counts > 0]
            if len(nan_with_values) > 0:
                f.write("  Features with NaN values (before imputation):\n")
                for col, count in nan_with_values.items():
                    f.write(f"    - {col}: {count} ({count/len(df)*100:.2f}%)\n")
            else:
                f.write("  No NaN values in HRV features (after median imputation)\n")
        
        print(f"[SAVED] Processing summary: {summary_path}")
        
        return {
            'hrv_features': hrv_path,
            'case_stats': stats_path,
            'summary': summary_path
        }


# ============================================================
# Main Execution
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract SCOPE ECG HRV features')
    parser.add_argument('--data_path', required=True,
                        help='Directory containing metadata_v1.0.xlsx and ECG files')
    parser.add_argument('--output_dir', required=True,
                        help='Directory to save processed features and statistics')
    parser.add_argument('--n_workers', type=int, default=N_WORKERS,
                        help='Number of parallel workers (default: 8)')
    parser.add_argument('--sequential', action='store_true',
                        help='Process cases without multiprocessing')
    args = parser.parse_args()

    # Initialize dataset processor
    dataset = SCOPEDataset(
        data_path=args.data_path,
        horizon_max_hour=HORIZON_MAX_HOUR,   # 24h upper bound
        window_min=WINDOW_MIN,               # 5-min epochs
        slide_min=SLIDE_MIN,                 # 5-min sliding step
        horizon_min_min=HORIZON_MIN_MIN,     # 30-min (0.5h) lower bound
        srate=TARGET_SRATE                   # 250 Hz after downsampling
    )

    # Process all cases with parallel execution
    df, stats_df = dataset.load_all_data(
        max_cases=MAX_CASES,
        use_parallel=USE_PARALLEL and not args.sequential, 
        n_workers=args.n_workers,
        specific_ids=SPECIFIC_IDS
    )

    # Save processed results
    saved_paths = dataset.save_results(df, stats_df, output_dir=args.output_dir)

    # Extract feature matrix for model training
    # Returns 43 HRV features as initial candidates for BorutaShap selection
    X, y, case_ids = dataset.get_feature_matrix(df)

    print("\n[INFO] Ready for model evaluation!")
