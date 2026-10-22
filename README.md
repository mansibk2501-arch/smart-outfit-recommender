# Smart Outfit Color Recommendation System

## Features
- Select a base clothing color from a palette.
- Upload a JPG/PNG image and estimate its dominant color using KMeans clustering.
- Rank matching colors using color-harmony rules plus a trained Random Forest classifier.
- Display held-out test metrics in the app.

## Windows setup (VS Code)
1. Extract this folder.
2. Open the folder in VS Code (`File > Open Folder`).
3. Open Terminal > New Terminal.
4. (Recommended) Create a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks activation, you can skip activation and use `python -m pip` commands.
5. Install dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```
6. Train the model and create the dataset + evaluation metrics:
   ```powershell
   python train_model.py
   ```
7. Start the app:
   ```powershell
   python -m streamlit run app.py
   ```
8. Open the `Local URL` shown in the terminal, usually `http://localhost:8501`.

## How to use
- Tab 1: choose a base color, occasion, and garment type; click Recommend colors.
- Tab 2: upload a clear clothing image and click Recommend from detected color.
- Tab 3: see test accuracy, precision, recall, confusion matrix, and classification report.

## Important limitation / honest ML explanation
The starter dataset is generated from a documented color-harmony heuristic and the model learns to approximate those rule-generated labels. The reported test metrics measure agreement with that heuristic, not expert or user-rated fashion compatibility. For a stronger research project, replace/augment the synthetic labels with human-rated outfit combinations and evaluate on those labels.

## Image detection limitation
KMeans detects the most common color in the whole uploaded image; it does not segment the garment from the background. Use a simple background and one prominent garment for the best demo. A future version can add clothing segmentation.

## Project files
- `app.py`: Streamlit user interface and recommendations.
- `color_utils.py`: color conversion, features, harmony scoring.
- `train_model.py`: dataset creation, model training, evaluation.
- `requirements.txt`: dependencies.
- `outfit_pairs_dataset.csv`: generated when training runs.
- `outfit_color_model.joblib`: generated trained model.
- `model_metrics.json`: generated evaluation results.
