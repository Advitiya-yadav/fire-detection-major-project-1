# Fire Detection from Aerial Images

Binary aerial-image classification: **Fire** or **Non-fire**. The architecture follows the workflow diagram supplied for the IEEE paper *Toward Intelligent Earth Observation: Hierarchical Feature Fusion and a Drone Dataset* by Danish et al.

## Live demo

[Open the Streamlit fire-detection app](https://advitiya-yadav-fire-detection-major-project-1-app-o7qwaz.streamlit.app/)

## Workflow

```text
66 online video sources
          ↓
Frame preprocessing, curation, resize to 224 × 224
          ↓
DFF — EfficientNetB2 fire-specific feature maps
          ↓
    ┌───────────────┴────────────────┐
    ↓                                ↓
FGA module                     Direct branch
7×7, 5×5, 3×3 attention       1×1 convolution
    └───────────────┬────────────────┘
                    ↓
          HFF hierarchical fusion
                    ↓
       BatchNorm → Global pooling
                    ↓
             FC(2) → Softmax
                    ↓
             Fire / Non-fire
```

The DFF feature maps split into parallel **FGA** and **direct 1×1-convolution** paths. Both paths feed into **HFF**, which fuses them before classification. Smoke is not a separate output class and is excluded from training; the classifier returns only **Fire** or **Non-fire**.

The repository uses an already-cleaned, manually organized dataset. It does not implement video collection or frame extraction; the 66-source collection and frame preprocessing boxes describe the upstream workflow shown in the supplied diagram.

## Dataset layout

Provide `train`, `val`, and `test` splits with `fire` and `nonfire` class folders:

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

Any `smoke` folders are ignored by the data loader. No dataset-merging step is required. Images are resized to 224 × 224. Training augmentation includes random horizontal flips, brightness, and contrast changes.

## Setup (Windows / Python 3.12)

From PowerShell in the project directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run commands without activating the environment by using `.\.venv\Scripts\python.exe` instead of `python`.

## Train

```powershell
python train.py --data-dir "C:\path\to\dataset"
```

The best checkpoint is saved to `training_output\best_model.keras` by default. Retrain after changing the architecture; weights from the former three-class model are incompatible with this binary model.

## Predict

Classify one image:

```powershell
python infer.py "C:\path\to\image.jpg"
```

Run the local web interface:

```powershell
streamlit run app.py
```

The app uses a local checkpoint when available; otherwise, it downloads the configured model from Hugging Face.

## Deployment

The Streamlit app is connected to this GitHub repository. A push to the branch selected in Streamlit Cloud normally triggers a rebuild/redeploy; check the Streamlit dashboard for the configured branch and deployment status.

Model weights are separate from code: `training_output` is gitignored. To deploy this updated binary architecture, train a compatible two-class model and update the model artifact at the configured Hugging Face source. A code push alone will not replace the hosted model, and the existing three-class hosted checkpoint is not compatible with this app.

## Main files

| File | Purpose |
| --- | --- |
| `model.py` | DFF backbone, FGA/direct branches, HFF fusion, and binary classifier |
| `data_pipeline.py` | Fire/nonfire data loading and preprocessing |
| `train.py` | Model training, fine-tuning, evaluation, and checkpoint saving |
| `infer.py` | Command-line prediction |
| `app.py` | Streamlit web application |
| `prediction.py` | Shared preprocessing and prediction utilities |
