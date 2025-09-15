Corn Stand Count Prototype

**Project Overview**
This prototype demonstrates a simple computer vision pipeline for training a model and counting corn plants in drone images.
The project goes from data preprocessing, model training, and evaluation to deployment in a Streamlit web app, packaged with Docker for portability.
An initial, rudimentary model is available for testing. It has been trained with 5 meter single rows.

**Repository Structure**
* stand_count/
  * app.py # Streamlit app for inference (UI)
  * requirements.txt # Python dependencies
  * README.md # Project documentation
  * models/ # Trained model(s) saved as .pth
  * src/ # Core source code
    * data_loader.py # Custom dataset and dataloader utilities
    * regression_model.py # CNN regression model + training loop
    * train_regression.py # Script for training the model
    * evaluate_regression.py # Model evaluation and plotting
  * data/ # CSVs + image folders
    * dataset_train.csv # Training metadata for new model training
    * dataset_test.csv # Test metadata for new model training
    * images/ # Image files for new model training

**Technologies Used**
Programming language: Python 3.10+
Deep Learning: PyTorch, TorchVision
Data handling: Pandas, NumPy, Pillow
Visualization: Matplotlib
Web app: Streamlit
Containerization: Docker

**Running Locally**
1. Clone the repository:
git clone https://github.com/yourusername/stand_count.git
cd stand_count
3. Create and activate a virtual environment:
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
4. Install dependencies:
pip install -r requirements.txt
5. Run the Streamlit app:
streamlit run app.py
6. Predictions will be saved in a CSV file in the root folder.

**Running with Docker**
1. Build the image:
docker build -t stand_count_app .
2. Run the container:
docker run -p 8501:8501 stand_count_app
3. Open the app in your browser:
http://localhost:8501
4. Predictions will be saved in a CSV file in the root folder.

**Training a New Model**
1. Copy training images to .data/images/train
2. Copy test images to .data/images/test
3. Set image labels to .data/dataset_train.csv and .data/dataset_test.csv
4. Run train_regression.py - the new model will be saved to .models

**Evaluating a model quality**
1. Open evaluate_model.py and paste the model name where prompted
2. Run the script, which will generate a chart comparing predicted and actual values
3. The script will also show MAE and RMSE

**Results**
The current model is simple (basic CNN) and may overestimate the number of plants.
The user can train a new model by including images in the provided folders and datasets.
The file plant_count_results.csv contains batch results (image name + prediction).
The main purpose is to demonstrate the full ML → Deployment workflow.

**Author Notes**
Developed as a learning project in computer vision and ML deployment.
An augmentation script is provided to create 5x new images in case the user has few samples to train a new model.




