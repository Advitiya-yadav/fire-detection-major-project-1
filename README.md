# Fire Detection from Aerial Images

This project classifies an image as **fire** or **nonfire** using a binary image classifier. It includes a TensorFlow training and evaluation workflow, a command-line predictor, and a Streamlit web app.

## Live demo

[Open the Streamlit fire-detection app](https://advitiya-yadav-fire-detection-major-project-1-app-o7qwaz.streamlit.app/)

## Model workflow

```text
224 × 224 RGB image
        ↓
EfficientNetB2 feature extractor (DFF)
        ↓
   ┌────┴──────────────────┐
   ↓                       ↓
FGA multiscale attention   Direct 1×1 convolution
7×7, 5×5, and 3×3          │
   └───────────┬───────────┘
               ↓
      HFF hierarchical fusion
               ↓
 Batch normalization → global average pooling
               ↓
        Dense(2) → softmax
               ↓
       fire / nonfire scores
```

The backbone starts with ImageNet-pretrained EfficientNetB2 weights. Training first freezes the backbone and can then fine-tune its final layers. Training augmentation applies random horizontal flips, brightness changes, and contrast changes. Predictions average the original and horizontally flipped image scores.

The model has two output classes: `fire` and `nonfire`. Smoke is not a separate class; any `smoke` directories are ignored by the standard folder-based loader. The architecture follows the DFF/FGA/HFF workflow described in the supplied diagram for Danish et al.'s paper *Toward Intelligent Earth Observation: Hierarchical Feature Fusion and a Drone Dataset*. Video collection and frame extraction are upstream activities and are not implemented in this repository.

## Dataset

The standard workflow expects `train`, `val`, and `test` directories, each containing `fire` and `nonfire` subdirectories:

```text
dataset/
├── train/
│   ├── fire/
│   └── nonfire/
├── val/
│   ├── fire/
│   └── nonfire/
└── test/
    ├── fire/
    └── nonfire/
```

The default data root is the repository's `merged_dataset` directory. The loader also accepts a parent directory that contains a nested `merged_dataset` folder. Images are resized to 224 × 224; the model's EfficientNetB2 backbone handles pixel normalization.

By default, all images in the `fire` and `nonfire` folders are used. To restrict training, validation, and testing to aerial images, pass `--aerial-only`. This mode requires a `manifest.csv` beside the split directories. The CSV must have `split`, `is_aerial`, `class`, and `dest_path` columns; `dest_path` is relative to the dataset directory, and `is_aerial` must be `true` for included rows. Smoke rows are skipped. This repository does not include a dataset-merging or manifest-generation script.

## Setup (Windows / Python 3.12)

In PowerShell, from the project directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If script execution is disabled in PowerShell, run commands with the environment's interpreter directly, for example:

```powershell
.\.venv\Scripts\python.exe train.py
```

## Train and evaluate

Train with the default dataset and settings:

```powershell
python train.py
```

Or choose a dataset root and output directory:

```powershell
python train.py --data-dir "C:\path\to\dataset" --output-dir "C:\path\to\training_output"
```

Training uses class weights, saves the best checkpoint by validation loss, and applies early stopping. It then calibrates a fire decision threshold using the validation nonfire images (default maximum false-positive rate: 1%), saves the threshold, and evaluates the best checkpoint on the held-out test split. The run reports test metrics, a per-class classification report, a confusion matrix, and the test false-positive rate at the calibrated fire threshold.

Useful options:

| Option | Default | Description |
| --- | ---: | --- |
| `--data-dir` | `merged_dataset` | Dataset root |
| `--output-dir` | `training_output` | Directory for model and calibration artifacts |
| `--epochs` | `15` | Initial training epochs |
| `--batch-size` | `32` | Batch size |
| `--learning-rate` | `0.0001` | Initial learning rate |
| `--patience` | `3` | Early-stopping patience |
| `--fine-tune-epochs` | `5` | Additional fine-tuning epochs; use `0` to disable |
| `--fine-tune-layers` | `30` | Final backbone layers to unfreeze |
| `--max-fire-fpr` | `0.01` | Maximum validation false-positive rate for fire threshold calibration |
| `--aerial-only` | Off | Use only manifest-marked aerial images |

The output directory contains `best_model.keras`, `best_model.weights.h5`, and `fire_threshold.json`. The `.keras` file is the complete model; the weights file can be loaded into the matching model architecture. These artifacts are excluded from Git by `.gitignore`.

## Predict

Classify an image using a local model if present, or download the configured Hugging Face model when the default local model is missing:

```powershell
python infer.py "C:\path\to\image.jpg"
```

Specify a model file or adjust the confidence review threshold:

```powershell
python infer.py "C:\path\to\image.jpg" --model "C:\path\to\best_model.keras" --min-confidence 0.60
```

The command prints the predicted class and both class scores. Fire predictions below the calibrated fire threshold are reported as uncertain. Other predictions below `--min-confidence` (default `0.60`) are also reported as uncertain. If no calibration file is available, inference uses a fallback fire threshold of `0.80`.

## Run the web app

```powershell
streamlit run app.py
```

Upload a supported image in the browser and select **Classify image**. The app first looks for `training_output\best_model.keras`; if it is missing, it downloads `best_model.keras` from the configured Hugging Face repository, `advitiya21/fire-detection-model`. It loads `fire_threshold.json` alongside the model when available. Make sure the hosted model is a compatible two-class model; an older three-class checkpoint is not compatible with the current app.

The displayed scores and threshold are not guarantees of real-world performance. Validation false-positive rates do not ensure the same rate on new images, and this project is not a safety system.

## Deployment

Streamlit Cloud deploys the repository and branch configured in its dashboard; check that dashboard for deployment status and settings. Model files in `training_output` are gitignored and are not deployed with the source. To publish a newly trained model, upload the compatible two-class `best_model.keras` and its corresponding `fire_threshold.json` to the configured Hugging Face repository. Updating application code alone does not update the hosted model.

## Project files

| File | Purpose |
| --- | --- |
| `model.py` | EfficientNetB2 backbone, FGA/direct branches, HFF fusion, and binary classifier |
| `data_pipeline.py` | Dataset discovery, preprocessing, augmentation, and class weighting |
| `train.py` | Training, optional fine-tuning, threshold calibration, and test evaluation |
| `prediction.py` | Shared image preprocessing, test-time augmentation, and threshold utilities |
| `infer.py` | Command-line image prediction |
| `app.py` | Streamlit image-classification app |
| `requirements.txt` | Python dependencies |
