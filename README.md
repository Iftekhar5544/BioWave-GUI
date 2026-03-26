# BioWave - EMG

BioWave - EMG is a desktop application for real-time EMG signal monitoring, guided data collection, and machine-learning based gesture classification. It is designed for end-to-end EMG workflow: connect and calibrate channels, capture labeled sessions, train Random Forest models, and run live predictions from trained artifacts.

## What This Project Provides

- Real-time multi-channel EMG visualization and analysis
- Channel calibration and signal quality tracking
- Guided task-based data collection with structured dataset output
- Configurable Random Forest training pipeline with saved run history
- Model loading for live inference in the main app
- Optional EMG simulator for development and testing without hardware

## Project Structure

- `code/` - Python source files
- `dataset/` - collected EMG datasets and recording bundles
- `trained_model/` - trained model bundles and training outputs
- `README.md`
- `requirements.txt`

## Run

From project root:

```powershell
python code/main.py
```

Optional simulator:

```powershell
python code/emg_simulator_app.py
```

## Notes

- New recordings are saved under `dataset/`.
- New training outputs and model artifacts are saved under `trained_model/`.
- The model loader defaults to `trained_model/`.
