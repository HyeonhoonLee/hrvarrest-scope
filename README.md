# hrvarrest-scope

ECG heart rate variability (HRV) extraction, LightGBM training with Beta
calibration, and evaluation for in-hospital cardiac arrest prediction on the
SCOPE dataset. The scripts extract features from five-minute ECG segments and
report discrimination, classification, and calibration metrics.

## Requirements

Python 3 with numpy, pandas, scipy, scikit-learn, lightgbm,
bayesian-optimization, betacal, matplotlib, neurokit2, vitaldb, tqdm,
openpyxl, pyarrow, and shap. Install these packages with:

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Inputs and usage

Run commands from the repository root. Place the SCOPE files in `data/SCOPE/`:
`metadata_v1.0.xlsx` with `case_id` and `outcome` columns, and one `.vital` file
named after each `case_id`. Outcomes are `Cardiac Arrest`, `Discharge`, and
`Death`; the processor uses the first two groups. It reads `ECG_II`, falling
back to another ECG track, and uses the dataset's cardiac arrest timestamp
convention (2100-01-01 00:00:00).

```sh
python3 dataprocessing/scope_dataset_processing.py \
  --data_path data/SCOPE --output_dir output/features

features=$(find output/features -type f -name scope_hrv_features.parquet | sort | tail -n 1)
sh modeling/train.sh \
  --data_path "$features" --output_dir output/training
```

Extraction writes a timestamped directory containing `scope_hrv_features.parquet`,
processing statistics, and a summary. The command above trains on the most
recent extraction. Training saves `best_model.txt`, `calibrator.pkl`, predictions,
metrics, and plots in `output/training/`.

For standalone evaluation, place the labeled HRV table to score at
`data/evaluation_hrv_features.parquet`. Feature tables require `case_id`, `label`
(1 for cardiac arrest, 0 for discharge), `time_to_event` in minutes, and the
`HRV_*` columns used by the model. Extraction also supplies `quality_ratio`.

```sh
sh modeling/evaluation.sh \
  --data_path data/evaluation_hrv_features.parquet \
  --model_path output/training/best_model.txt --output_dir output/evaluation
```

Training evaluates Beta-calibrated predictions. Standalone evaluation uses raw
LightGBM probabilities, does not load `calibrator.pkl`, and selects its
classification threshold using the supplied evaluation labels.

## Data

The SCOPE dataset is available from KHDP under its own terms. Data and fitted
models are not included in this repository.

SCOPE: A survival and cardiac arrest outcome dataset with PPG and ECG waveforms
from intensive care units (version 1.0). KHDP; 2025.
doi:[10.23421/2025.0698](https://doi.org/10.23421/2025.0698).

## Related work

The method follows Lee H et al., Real-time machine learning model to predict
in-hospital cardiac arrest using heart rate variability in ICU. *npj Digit Med*.
2023;6:215. doi:[10.1038/s41746-023-00960-2](https://doi.org/10.1038/s41746-023-00960-2).
Its code is available at [hrvarrest](https://github.com/HyeonhoonLee/hrvarrest)
and is not included here.

## Licence

Apache-2.0. See [LICENSE](LICENSE) for the full text. The code licence does not
change the dataset's terms.
