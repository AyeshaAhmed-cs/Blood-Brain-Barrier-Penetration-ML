import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator

# Disable RDKit logs
RDLogger.DisableLog('rdApp.*')

# ==============================================================================
# 1. PAGE CONFIGURATION & DARK THEME STYLING
# ==============================================================================
st.set_page_config(
    page_title="BBB Permeability Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor': '#0e1117',
    'axes.facecolor': '#0e1117',
    'text.color': '#ffffff',
    'axes.labelcolor': '#ffffff',
    'xtick.color': '#ffffff',
    'ytick.color': '#ffffff'
})

# ==============================================================================
# 2. ASSET & DATA LOADING
# ==============================================================================
@st.cache_resource
def load_assets():
    model = joblib.load('bbb_penetration_rf_model.pkl')
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    return model, mfpgen

@st.cache_data
def load_dataset():
    df = pd.read_csv('BBBP.csv')
    df_clean = df.dropna(subset=['smiles']).copy()
    
    # physical descriptors for visualizations
    mws, logps, tpsas, hbds, hbas, rots = [], [], [], [], [], []
    for s in df_clean['smiles']:
        mol = Chem.MolFromSmiles(s)
        if mol:
            mws.append(Descriptors.MolWt(mol))
            logps.append(Descriptors.MolLogP(mol))
            tpsas.append(Descriptors.TPSA(mol))
            hbds.append(Descriptors.NumHDonors(mol))
            hbas.append(Descriptors.NumHAcceptors(mol))
            rots.append(Descriptors.NumRotatableBonds(mol))
        else:
            mws.append(None)
            logps.append(None)
            tpsas.append(None)
            hbds.append(None)
            hbas.append(None)
            rots.append(None)
            
    df_clean['Molecular_Weight'] = mws
    df_clean['LogP'] = logps
    df_clean['TPSA'] = tpsas
    df_clean['H_Bond_Donors'] = hbds
    df_clean['H_Bond_Acceptors'] = hbas
    df_clean['Rotatable_Bonds'] = rots
    df_clean['BBB_Status'] = df_clean['p_np'].map({1: 'Permeable (1)', 0: 'Non-permeable (0)'})
    return df_clean.dropna()

model, mfpgen = load_assets()

# ==============================================================================
# 3. SIDEBAR & TEST MOLECULES
# ==============================================================================
st.sidebar.title("🧬 Test Suite & Info")
st.sidebar.markdown("Select a sample compound or enter custom SMILES:")

test_molecules = {
    "Custom Input": "",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Dopamine": "C1=CC(=C(C=C1CCN)O)O",
    "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "Penicillin G": "CC1(C(N2C(S1)C(C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C"
}

selected_option = st.sidebar.selectbox("Preset Compounds:", list(test_molecules.keys()))

# Set default_smiles to empty string if Custom Input is picked
if selected_option != "Custom Input":
    default_smiles = test_molecules[selected_option]
else:
    default_smiles = ""

st.sidebar.markdown("---")
st.sidebar.info("""
**Model Specs:**
- **Algorithm:** Random Forest Classifier
- **Features:** 1024-bit Morgan Fingerprints + 6 Physical Descriptors
- **ROC-AUC:** 0.9518
""")

# ==============================================================================
# 4. MAIN INTERFACE & TABS
# ==============================================================================
st.title("🧠 Blood-Brain Barrier Penetration Dashboard")
st.markdown("Predict molecular permeability across the blood-brain barrier using Machine Learning & RDKit.")

tab1, tab2 = st.tabs(["🧪 Predictor Workspace", "📊 Exploratory Data Analysis"])

# ------------------------------------------------------------------------------
# TAB 1: PREDICTOR WORKSPACE
# ------------------------------------------------------------------------------
with tab1:
    col_input, col_output = st.columns([1, 1.2], gap="medium")

    with col_input:
        st.subheader("📥 Input SMILES")
        smiles = st.text_input(
            "Enter Molecule SMILES String:", 
            value=default_smiles,
            placeholder="Paste SMILES string here (e.g., CC(=O)OC1=CC=CC=C1C(=O)O)..."
        )
        predict_btn = st.button("🔮 Predict Permeability", use_container_width=True, type="primary")

    with col_output:
        st.subheader("🎯 Prediction Output")
        
        if predict_btn:
            if not smiles.strip():
                st.warning("⚠️ Please enter or paste a SMILES string first!")
            else:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    st.error("❌ Invalid SMILES string! Please check the structure and try again.")
                else:
                    # Compute Features
                    fp_np = np.array(mfpgen.GetFingerprint(mol))
                    mw = Descriptors.MolWt(mol)
                    logp = Descriptors.MolLogP(mol)
                    tpsa = Descriptors.TPSA(mol)
                    hbd = Descriptors.NumHDonors(mol)
                    hba = Descriptors.NumHAcceptors(mol)
                    rot = Descriptors.NumRotatableBonds(mol)

                    phys_desc = np.array([mw, logp, tpsa, hbd, hba, rot])
                    features = np.hstack((fp_np, phys_desc)).reshape(1, -1)

                    prob = model.predict_proba(features)[0][1]
                    pred = model.predict(features)[0]

                    # Status Banner
                    if pred == 1:
                        st.success(f"### Result: Permeable (Crosses BBB)")
                    else:
                        st.error(f"### Result: Non-permeable (Blocked)")

                    # Probability Gauge
                    st.write(f"**Permeability Probability:** `{prob * 100:.2f}%`")
                    st.progress(float(prob))

                    st.markdown("---")
                    st.markdown("#### 📐 Calculated Molecular Properties")
                    m_col1, m_col2, m_col3 = st.columns(3)
                    m_col1.metric("Mol. Weight", f"{mw:.1f} g/mol")
                    m_col2.metric("LogP (Lipophilicity)", f"{logp:.2f}")
                    m_col3.metric("TPSA", f"{tpsa:.1f} Å²")

                    m_col4, m_col5, m_col6 = st.columns(3)
                    m_col4.metric("H-Donors", hbd)
                    m_col5.metric("H-Acceptors", hba)
                    m_col6.metric("Rotatable Bonds", rot)
        else:
            st.info("👈 Select a preset or paste any custom SMILES string from your dataset, then click **Predict Permeability**.")

# ------------------------------------------------------------------------------
# TAB 2: EXPLORATORY DATA ANALYSIS (NOTEBOOK PLOTS)
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("📊 Dataset Visualizations & Chemical Space Analysis")
    
    try:
        df_clean = load_dataset()

        eda_col1, eda_col2 = st.columns([1, 1], gap="large")

        with eda_col1:
            st.markdown("#### Target Class Distribution")
            fig1, ax1 = plt.subplots(figsize=(6, 4.5))
            sns.countplot(
                data=df_clean, x='BBB_Status', hue='BBB_Status', 
                palette=['#4a90e2', '#e74c3c'], ax=ax1, legend=False
            )
            ax1.set_title('Target Class Distribution', fontsize=12, fontweight='bold')
            ax1.set_xlabel('')
            ax1.set_ylabel('Count')
            plt.tight_layout()
            st.pyplot(fig1)

        with eda_col2:
            st.markdown("#### Chemical Properties Correlation")
            fig2, ax2 = plt.subplots(figsize=(7, 4.5))
            corr = df_clean[['Molecular_Weight', 'LogP', 'TPSA', 'H_Bond_Donors', 'H_Bond_Acceptors', 'Rotatable_Bonds', 'p_np']].corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap='mako', ax=ax2, cbar=True)
            ax2.set_title('Correlation Heatmap')
            plt.tight_layout()
            st.pyplot(fig2)

        st.markdown("---")
        st.markdown("#### Chemical Space Exploration: TPSA vs LogP")
        fig3, ax3 = plt.subplots(figsize=(10, 4.5))
        sns.scatterplot(
            data=df_clean, x='TPSA', y='LogP', hue='BBB_Status',
            alpha=0.8, palette=['#4a90e2', '#e74c3c'], ax=ax3
        )
        ax3.set_title('Chemical Space: Topological Polar Surface Area vs Lipophilicity', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig3)

    except FileNotFoundError:
        st.warning("⚠️ `BBBP.csv` file not found in the root directory. Place `BBBP.csv` alongside `app.py` to view dataset analytics.")
