🧠 Blood-Brain Barrier (BBB) Penetration Predictor & Dashboard
An end-to-end Machine Learning web application and exploratory data analysis pipeline built with RDKit, Scikit-Learn, and Streamlit to predict whether chemical compounds can cross the Blood-Brain Barrier based on their SMILES representations.

🚀 Live Interactive App: [Launch Streamlit Dashboard](http://localhost:8501/)

📌 Project Overview
The Blood-Brain Barrier (BBB) acts as a highly selective semipermeable border that protects the central nervous system. Accurately predicting whether a drug molecule can penetrate this barrier is a critical step in neuropharmaceutical drug discovery.

This repository contains a complete pipeline ranging from molecular descriptor feature extraction and Exploratory Data Analysis (EDA) to a trained Random Forest classification model served through a fully styled interactive Streamlit dashboard.

✨ Features & Architecture
Interactive Predictor Workspace: Paste any custom SMILES string or choose from preset compounds (Caffeine, Aspirin, Dopamine, Ibuprofen, Penicillin G) to instantly evaluate permeability probability.

Advanced Molecular Feature Engineering: Automatically computes 1024-bit Morgan Fingerprints (Radius 2) combined with 6 key physical chemical properties using RDKit:

Molecular Weight (MW)

Lipophilicity (LogP)

Topological Polar Surface Area (TPSA)

Hydrogen Bond Donors (HBD)

Hydrogen Bond Acceptors (HBA)

Rotatable Bonds

Exploratory Data Analysis (EDA) Tab: Visualizes dataset distributions, chemical property correlation heatmaps, and chemical space scatter plots (TPSA vs LogP).

Optimized Model: Trained via a Random Forest Classifier achieving strong predictive performance (0.9518 ROC-AUC).

🛠️ Tech Stack
Programming Language: Python

Machine Learning & Chemoinformatics: Scikit-Learn, RDKit, Joblib, NumPy, Pandas

Data Visualization: Matplotlib, Seaborn

Web Framework: Streamlit

📂 Repository Directory Structure
Plaintext
blood-brain-barrier-penetration/
│
├── BBBP (1).ipynb                 # Jupyter Notebook detailing model training & EDA
├── BBBP.csv                       # Molecular dataset containing SMILES and targets
├── README.md                      # Project documentation
├── app.py                         # Main Streamlit dashboard script
└── bbb_penetration_rf_model.pkl   # Serialized Random Forest model weights

⚙️ Installation & Local Execution
To run this project locally on your machine, follow these steps:

Clone the repository:

Bash
git clone https://github.com/AyeshaAhmed-cs/Blood-Brain-Barrier-Penetration-ML
cd blood-brain-barrier-penetration
Install the required dependencies:

Bash
pip install streamlit scikit-learn rdkit pandas numpy joblib matplotlib seaborn
Launch the Streamlit dashboard:

Bash
streamlit run app.py
