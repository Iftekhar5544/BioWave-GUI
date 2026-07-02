# BioWave - EMG

BioWave - EMG is a Python desktop application for real-time EMG monitoring, guided dataset collection, Random Forest training, and live gesture classification. The current project supports both a development simulator and ESP32-S3 based hardware that streams 8 EMG channels plus BNO080 IMU orientation data.

The application is built around a complete workflow:

1. Connect to a wired serial stream, TCP simulator stream, or wireless ESP32 stream.
2. Calibrate rest and flex baselines for the active channels.
3. Monitor live EMG/IMU plots and signal quality metrics.
4. Run guided recording protocols for labeled gestures.
5. Save structured dataset bundles under `dataset/`.
6. Train Random Forest models from selected CSV datasets.
7. Load trained model artifacts and run live inference in the main app.

## Project Layout

```text
EMG_with_IMU/
|-- esp_code_emg_imu.txt
|-- archive/
|   |-- app.py
|   |-- esp_code.txt
|   `-- recordings/
`-- 01_BioWave-EMG Data Collection APP/
    |-- README.md
    |-- requirements.txt
    |-- code/
    |   |-- main.py
    |   |-- rf_features.py
    |   |-- train_rf_model_gui.py
    |   |-- emg_simulator_app.py
    |   |-- app_theme.py
    |   `-- images/app_icon.png
    |-- dataset/
    `-- trained_model/
```

## Code Guide

- `code/main.py` is the main BioWave desktop app. It contains connection dialogs, serial and wireless stream workers, calibration, live plotting, analysis windows, guided data collection, RF training, model loading, and real-time inference.
- `code/rf_features.py` defines the shared Random Forest feature contract used by training and live prediction. It extracts time-domain, spectral, RMS-ratio, and pairwise-correlation features from windowed EMG data.
- `code/emg_simulator_app.py` creates synthetic EMG streams for development. TCP mode is the easiest way to test the main app without hardware.
- `code/train_rf_model_gui.py` is a smaller standalone trainer. It is useful for older 4-channel CSV workflows, but the integrated trainer in `main.py` is the current path for variable-channel datasets.
- `code/app_theme.py` centralizes the PyQt dark theme, button styles, label styles, and Windows title-bar styling.
- `esp_code_emg_imu.txt` is the current ESP32-S3 firmware source text for Wi-Fi discovery/control, USB provisioning, 8-channel EMG sampling, BNO080 IMU polling, and UDP streaming.
- `archive/` contains older app, firmware, and recording artifacts kept for reference.

## Requirements

The Python dependencies are listed in `requirements.txt`:

```text
numpy
pyqt5
pyqtgraph
pyserial
joblib
scikit-learn
PyWavelets
```

Recommended setup from the app folder:

```powershell
cd "01_BioWave-EMG Data Collection APP"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `python` is not available on your Windows PATH, install Python and enable the PATH option, or use your configured Python executable.

## Running The App

From `01_BioWave-EMG Data Collection APP/`:

```powershell
python code/main.py
```

Optional simulator:

```powershell
python code/emg_simulator_app.py
```

For simulator testing:

1. Start `code/emg_simulator_app.py`.
2. Choose `TCP Server (Recommended)`.
3. Start the simulator.
4. Open `code/main.py`.
5. In port configuration, use:

```text
socket://127.0.0.1:7000
```

## Hardware And Streaming

The current firmware target is an ESP32-S3 with:

- 8 EMG analog channels sampled at 500 Hz.
- BNO080 fused orientation output for roll, pitch, and yaw.
- USB serial provisioning for Wi-Fi credentials.
- UDP discovery/control on port `5001`.
- UDP data streaming on port `5000`.
- HMAC-authenticated wireless control commands.

The wireless packet format uses a `BWIM` binary header followed by 5 frames per UDP packet. Each frame contains EMG values, IMU sample metadata, roll, pitch, yaw, and an IMU freshness flag. The main app includes a `WirelessStreamWorker` parser for this combined EMG plus IMU format.

Before flashing firmware, change:

```cpp
static const char *DEVICE_ACCESS_KEY = "CHANGE_THIS_TO_A_LONG_RANDOM_KEY";
```

The same access key must be entered in the desktop app when connecting wirelessly.

## Data Collection

New recording bundles are saved under `dataset/`. A typical bundle contains:

```text
dataset/<contributor>_<timestamp>/
|-- emg_data_<contributor>_<timestamp>.csv
`-- metadata_<contributor>_<timestamp>.txt
```

The current wireless dataset CSV layout is:

```text
Timestamp_ms,Packet_Number,Trial_ID,Ch1,Ch2,Ch3,Ch4,Ch5,Ch6,Ch7,Ch8,Ch9,Ch10,Ch11,Label
```

For wireless recordings:

- `Ch1..Ch8` are EMG channels.
- `Ch9..Ch11` are IMU orientation values.
- `Label` is the task or gesture label used for training.

Metadata records contributor details, selected labels, timing settings, channel count, connection medium, sample rate, expected sample count, actual sample count, and sample coverage.

## Random Forest Training

Training outputs are saved under `trained_model/` as run folders:

```text
trained_model/rf_training_<timestamp>/
|-- rf_realtime_model.joblib
|-- training_setup.json
|-- training_results.json
|-- training_summary.txt
|-- classification_report.txt
`-- confusion_matrix.csv
```

The integrated trainer in `main.py` supports multiple selected dataset CSV files as long as they have the same active channel count. It saves model metadata needed for live inference, including:

- Class names.
- Input channel count.
- Window and stride samples.
- Feature names.
- Sample rate.
- Training setup and evaluation results.

The integrated trainer defaults are tuned for responsive real-time inference:

- Trees (`n_estimators`): 150
- Max depth: 16
- Minimum samples per leaf: 2

These give fast live predictions at minimal accuracy cost compared to larger forests, and all three are adjustable in the training dialog.

The trainer uses a group-aware train/test split keyed to each source recording, so overlapping windows from the same recording never straddle the train/test boundary. This removes the leakage that would otherwise inflate reported accuracy. When a dataset is too small for a grouped split to keep every class on both sides, it falls back to a stratified split.

Feature extraction across windows is parallelized (joblib) for larger datasets.

Run folders are timestamped, for example `trained_model/rf_training_<timestamp>/`. Each `training_setup.json` records the window/stride, hyperparameters, channel count, and dataset paths used for that run.

## Feature Extraction

`rf_features.py` is the model feature contract. Keep training and inference aligned with this file. For each channel, it computes:

- Mean absolute value.
- RMS.
- Integrated EMG.
- Variance.
- Waveform length.
- Zero crossings.
- Slope sign changes.
- Willison amplitude.
- Mean, median, and peak frequency.
- Spectral entropy.
- Band-power percentages for 20-60 Hz, 60-120 Hz, and 120-220 Hz.

It also adds RMS-ratio features and pairwise channel-correlation features.

For `n` channels, the expected RF feature count is:

```text
n * 15 + n + (n * (n - 1) / 2)
```

For 11 channels this produces 231 features.

Features are computed with a vectorized implementation that processes all channels at once. The original per-channel implementation is retained in the same file as `_extract_window_features_reference`, and the two are verified to produce bit-identical output. If you ever change `rf_features.py`, keep the two in sync (diff the fast path against the reference) so existing trained models stay compatible.

## Typical Workflow

1. Start the app with `python code/main.py`.
2. Open port configuration.
3. Connect to simulator, wired serial, or wireless ESP32.
4. Set the channel count if needed.
5. Run channel calibration.
6. Open data collection.
7. Configure labels, repeats, prep, hold, and rest timing.
8. Record a task protocol.
9. Train a Random Forest model from one or more saved CSVs.
10. Load the generated `rf_realtime_model.joblib`.
11. Open real-time classification and monitor predictions.

## Current Repository State

At the time this README was updated, the workspace contains:

- 5 Python source files in `code/`.
- 14 dataset bundles under `dataset/`.
- 6 trained model runs under `trained_model/`.
- A current ESP32-S3 EMG plus IMU firmware text file at the repository root.
- Archived legacy app, firmware, and recording files under `archive/`.

Dataset and trained-model counts drift as you record and train; treat the numbers above as a snapshot.

## Maintenance Notes

- `main.py` is the primary application and is large. Future refactors should split hardware IO, recording, training, and analysis into smaller modules.
- Prefer the integrated training UI in `main.py` for current 8 EMG plus 3 IMU recordings.
- Treat `rf_features.py` as a compatibility boundary. Changing it requires retraining models. Keep the vectorized path and `_extract_window_features_reference` bit-identical.
- Real-time behavior is tuned for responsiveness: plot antialiasing is off by default (see `ENABLE_ANTIALIAS`/`ENABLE_OPENGL`), plots use peak downsampling and clip-to-view, live inference is capped at `RF_MAX_PREDICTION_HZ` with an in-flight guard and majority-vote label smoothing, and the Live Analysis window splits cheap metrics from the expensive correlation/coherence matrices (`LIVE_ANALYSIS_HEAVY_REFRESH_MS`). These knobs are constants near the top of `main.py`.
- `requirements.txt` is currently unpinned. Pin versions if this app needs reproducible installs.
- `__pycache__/`, generated datasets, and trained `.joblib` model artifacts are runtime outputs. Decide deliberately whether to keep them in version control.
- Keep firmware access keys private and never commit real Wi-Fi credentials or production access keys.
