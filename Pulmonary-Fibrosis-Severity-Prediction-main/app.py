import streamlit as st
import joblib
import torch
import torch.nn as nn

# Define the FVCTransformer class
class FVCTransformer(nn.Module):
    def __init__(self, fvc_input_dim, output_dim):
        super(FVCTransformer, self).__init__()
        self.fvc_linear = nn.Linear(fvc_input_dim, 64)
        self.output_linear = nn.Linear(64, output_dim)

    def forward(self, fvc):
        fvc_output = self.fvc_linear(fvc)
        output = self.output_linear(fvc_output)
        return output

# Load the trained model
model = joblib.load("fvc_transformer_model.pkl")

# === Rule-based severity function ===
def classify_severity(predicted_fvc, baseline_fvc):
    """
    Classifies severity based on ratio of predicted FVC to baseline FVC.
    """
    ratio = predicted_fvc / baseline_fvc
    if ratio >= 0.8:
        return "Mild"
    elif 0.6 <= ratio < 0.8:
        return "Moderate"
    else:
        return "Severe"

# ... (imports and model loading remain the same)

st.set_page_config(page_title="Pulmonary Fibrosis Prediction", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
        .big-font {
            font-size:20px !important;
        }
        .fvc-box {
            background-color: #1c1c1c;
            border-left: 5px solid #28a745;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            font-family: monospace;
            color: #00FF7F;
        }
        .severity-box {
            background-color: #282828;
            border: 2px dashed #1E90FF;
            padding: 12px;
            border-radius: 6px;
            text-align: center;
            font-size: 22px;
            font-weight: bold;
            color: #1E90FF;
        }
    </style>
""", unsafe_allow_html=True)

# Page Title and Description
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>Pulmonary Fibrosis Prediction</h1>
    <p style='text-align: center; color: #BBBBBB;'>An intelligent assistant that predicts FVC and classifies fibrosis severity using AI.</p>
    """,
    unsafe_allow_html=True,
)

# Input Section
st.subheader("📥 Enter Patient FVC Values")
with st.form("input_form"):
    fvc1 = st.text_input("FVC Value 1:")
    fvc2 = st.text_input("FVC Value 2:")
    fvc3 = st.text_input("FVC Value 3:")
    fvc4 = st.text_input("FVC Value 4:")
    fvc5 = st.text_input("FVC Value 5:")
    baseline_fvc = st.text_input("Baseline FVC (for severity classification):")
    submitted = st.form_submit_button("🔍 Predict")

# Handle prediction
if submitted:
    try:
        fvc_values = [float(fvc1), float(fvc2), float(fvc3), float(fvc4), float(fvc5)]
        input_tensor = torch.tensor(fvc_values, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            prediction = model(input_tensor)

        prediction_list = prediction.squeeze(0).tolist()
        predicted_fvc = prediction_list[-1]

        st.success("✅ Prediction Successful!")

        # Display predictions
        st.markdown("#### 📊 Predicted FVC Values")
        st.markdown('<div class="fvc-box">{}</div>'.format("<br>".join(
            [f"{i + 1}. {round(val, 2)}" for i, val in enumerate(prediction_list)]
        )), unsafe_allow_html=True)

        # Display severity
        if baseline_fvc:
            severity = classify_severity(predicted_fvc, float(baseline_fvc))
            st.markdown('<div class="severity-box">Predicted Pulmonary Fibrosis Severity: <span style="color:#FFD700;">{}</span></div>'.format(severity), unsafe_allow_html=True)
        else:
            st.warning("⚠️ Severity classification requires baseline FVC.")

    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")

# Divider and Info
st.markdown("---")
st.subheader("🧠 How It Works")
st.markdown("""
The model uses your recent FVC values to forecast lung capacity and estimate disease severity.
It’s trained on a **20GB dataset** combining:
- Historical FVC values (tabular data)
- CT scan images (visual features)

These multimodal insights enable **highly accurate predictions**.
""")

st.markdown("### ✨ Key Features")
st.markdown("""
- 📈 Predicts future FVC from **5 historical values**.
- 🧠 Uses deep learning on **tabular + CT scan** data.
- ✅ Severity is **automatically classified** (Mild, Moderate, Severe).
- 👨‍⚕️ User-friendly interface for medical practitioners and researchers.
""")

# Footer
st.markdown("---")
st.markdown(
    """
    <p style='text-align: center; color: #888;'>🔗 View source code on 
    <a href="https://github.com/UjjawalSah/Pulmonary-Fibrosis-Severity-Prediction" target="_blank" style="color: #4CAF50;">GitHub</a></p>
    """,
    unsafe_allow_html=True,
)
