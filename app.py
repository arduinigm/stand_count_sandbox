# app.py
# Basic Streamlit interface to predict plant number on an image of a corn single row

import streamlit as st
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
import os
from datetime import datetime
from src.regression_model import CornCNN

st.set_page_config(page_title="Corn Stand Counter - Single Rows", layout="centered")

MODEL_DIR = "models"


def get_latest_model(model_dir=MODEL_DIR):
    """
    Loads the latest model
    """
    models = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".pth")]
    if not models:
        return None
    return max(models, key=os.path.getctime)

model_path = get_latest_model()

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CornCNN()
if model_path:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
else:
    st.warning("No models available in src/models - a model needs to be trained first.")

# Transformation list
transform = transforms.Compose([
    transforms.Resize((100, 1000)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Interface
st.title("🌽 Corn Stand Counter - Single Rows")
st.write("Upload a picture of a Single Row corn plot:")

uploaded_files = st.file_uploader("Select files", type=["jpg", "jpeg", "png"], accept_multiple_files=True)


if uploaded_files:
    results = []

    for idx, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)

        # Prediction
        with torch.no_grad():
            output = model(img_tensor)
            predicted = int(round(output.item()))

        # Save results
        results.append({
            "image_name": uploaded_file.name,
            "number_of_plants": predicted
        })

        # Shows first image as sample
        if idx == 0:
            st.image(image, caption=f"Image: {uploaded_file.name}", use_column_width=True)
            st.write(f"🌱 1st image prediction: **{predicted} plants**")

    # Convert to pd.df and save
    df_results = pd.DataFrame(results)
    csv_path = "plant_count_results.csv"
    df_results.to_csv(csv_path, index=False)

    st.success(f"All images processed. Results saved to: {csv_path}.")
    st.dataframe(df_results)