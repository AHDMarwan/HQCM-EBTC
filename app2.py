import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from model import HQNetPL
import base64
import os
import hashlib
import json
from sklearn.calibration import calibration_curve
from cryptography.fernet import Fernet
from fpdf import FPDF
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import datetime
import umap
import joblib

# ------------------------
# 🔧 Model & Metadata
# ------------------------
n_qubits = 5
circuit_depth = 2
n_parallel_circuits = 5
class_number = 4
class_names = [
    "No Tumor (0)",
    "Meningioma (1)",
    "Glioma (2)",
    "Pituitary Tumor (3)"
]

# ------------------------
# 🌐 Streamlit UI Setup
# ------------------------
st.set_page_config(
    page_title="Quantum Brain Tumor Classifier",
    layout="wide",
    page_icon="🧠"
)

# Custom CSS for theme
st.markdown("""<style>
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #8e44ad, #3498db);
    }
    [data-testid="stMetric"] {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 15px;
        color:#ffffff;
    }
    .stAlert {
        border-left: 5px solid #8e44ad;
    }
    .stButton button {
        background-color: #3498db;
        color: white;
        font-size: 14px;
    }
    .stButton button:hover {
        background-color: #2c3e50;
    }
    .stSidebar {
        background-color: #2c3e50;
        color: #FFFFFF;
    }
</style>""", unsafe_allow_html=True)

st.title("🧠 Quantum-Classical Brain Tumor Classifier")
st.caption("Leveraging Quantum-enhanced Models to Accurately Classify Brain Tumor Types from MRI Scans")
with st.expander("📚 About | Brain Tumor Awareness & Support", expanded=False):
    st.markdown("""
    ## 🧠 About This Application
    This application uses a **Quantum-Classical Hybrid Neural Network** to classify MRI brain scans into four categories:
    
    - **No Tumor**
    - **Meningioma**
    - **Glioma**
    - **Pituitary Tumor**
    
    The model combines traditional deep learning with quantum computing techniques to enhance diagnostic accuracy and interpretability. It aims to support radiologists, clinicians, and researchers in identifying brain tumors more effectively.

    ---

    ## 🧩 Symptoms and Signs of Brain Tumors
    Recognizing early symptoms can be critical. Common signs include:
    
    - Persistent or worsening **headaches**
    - **Seizures** or convulsions
    - Changes in **vision**, such as blurred or double vision
    - **Difficulty speaking** or understanding language
    - Sudden **personality changes** or mood swings
    - Problems with **balance**, **coordination**, or **motor skills**

    It's important to note that symptoms may vary depending on the tumor's location and size.

    ---

    ## 🩺 How to Monitor Brain Health & When to Seek Help
    Proactive health monitoring plays a key role in early detection:
    
    - Schedule **regular medical check-ups** and **neuroimaging scans** if you're at higher risk.
    - **Document** and track any unusual neurological changes or symptoms.
    - **Seek immediate medical attention** if symptoms persist or worsen over time.
    - Early diagnosis and treatment significantly improve health outcomes and recovery prospects.

    ---

    ## 🖼️ Sample MRI Images
    For reference or experimentation, you can download open-source sample brain MRI datasets from:
    
    - [Kaggle Brain Tumor Dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
    - [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/)
    - [OpenNeuro Datasets](https://openneuro.org/)

    These resources offer diverse, labeled data suitable for both testing and research purposes.

    ---
    **Disclaimer:** This tool is intended for educational and research use only. It is **not a substitute for professional medical diagnosis or treatment.** Always consult a licensed healthcare provider for concerns about brain health.
    """)


# Sidebar with expanded sections for better organization
with st.sidebar:
    st.markdown("## 📤 Upload MRI Image")
    uploaded_file = st.file_uploader(
        "Upload a grayscale MRI image",
        type=['png', 'jpg', 'jpeg'],
        help="For best results, upload images of size 128x128 or 256x256."
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("L")
        st.image(image.resize((128, 128)), caption="Uploaded MRI Image", use_container_width=True)
    
    import pandas as pd

    # Example data for bar chart visualization
    performance_data = {
        'Metric': ['Precision', 'Recall', 'F1-Score'],
        'No Tumor': [0.99, 0.97, 0.98],
        'Meningioma': [0.96, 0.91, 0.93],
        'Glioma': [0.94, 0.98, 0.96],
        'Pituitary Tumor': [0.96, 0.99, 0.97],
        'Macro Avg': [0.96, 0.96, 0.96],
        'Weighted Avg': [0.96, 0.96, 0.96]
    }

    # Convert the data into a pandas DataFrame for easy plotting
    df_performance = pd.DataFrame(performance_data)

    # Plot the bar chart
    st.markdown("## 📈 Model Performance (HQCM-EBTC)")
    st.metric("Overall Accuracy (HQCM-EBTC)", "96.48%")

    # Show a performance bar chart
    st.markdown("### Detailed Performance Breakdown")
    st.bar_chart(df_performance.set_index('Metric').T)


# ------------------------
# 🧠 Model Initialization
# ------------------------

@st.cache_resource
def load_model():
    model = HQNetPL(
        n_qubits=n_qubits,
        circuit_depth=circuit_depth,
        n_parallel_circuits=n_parallel_circuits,
        class_number=class_number
    )
    checkpoint = torch.load(
        "HQNetPL_checkpoint_2025-03-28_22-08-26.pth",
        map_location='cpu'
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

model = load_model()

# ------------------------
# 📈 Inference Pipeline
# ------------------------
if uploaded_file:
    # Preprocessing for consistency and quality control
    transform = transforms.Compose([ 
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    try:
        img_tensor = transform(image).unsqueeze(0)
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.stop()
    
    # Model Inference
    with torch.no_grad():
        features = model.feature_extractor(img_tensor)
        attention_map = model.attention(features).squeeze(0)
        output = model(img_tensor)
        probs = F.softmax(output, dim=1).squeeze().numpy()
        pred_class = int(np.argmax(probs))
        confidence = probs[pred_class]

        # ------------------------
        # 🔧 Quantum Layer Output Extraction
        # ------------------------

        x = features * (1 + attention_map)  # Normalize attention map to [0, 1] by element-wise multiplication

        # Flatten and apply classical layers (like a fully connected layer)
        x = x.view(x.size(0), -1)
        x = model.dropout(x)
        x = F.relu(model.fc1(x))  # Apply ReLU activation after dropout and fully connected layer

        # Split into parallel quantum layers (assuming a model with parallel quantum circuits)
        n_parallel = len(model.qlayers)  # Number of parallel quantum layers
        x_split = torch.chunk(x, n_parallel, dim=1)  # Split input tensor across parallel layers

        # Process the features through the quantum layers
        quantum_outputs = torch.stack(
            [layer(x_i) for layer, x_i in zip(model.qlayers, x_split)], dim=1
        )

        # Reshape quantum outputs and convert to numpy array for further processing (e.g., UMAP)
        quantum_outputs = quantum_outputs.view(x.size(0), -1).numpy()


    # Attention Processing
    attention_avg = attention_map.mean(0).numpy()
    attention_resized = F.interpolate(
        torch.tensor(attention_avg).unsqueeze(0).unsqueeze(0),
        size=(128, 128),
        mode='bilinear',
        align_corners=False
    ).squeeze().numpy()
    attention_norm = (attention_resized - attention_resized.min()) / \
                    (attention_resized.max() - attention_resized.min() + 1e-6)

    # ------------------------
    # 📊 Results Display
    # ------------------------
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Prediction Probabilities")
        st.write("This section displays the predicted probabilities for each tumor type, which indicate the likelihood of the image belonging to each class. These values help us understand how confident the model is about the classification.")
        sorted_probs = sorted(zip(class_names, probs), key=lambda x: x[1], reverse=True)
        st.table({
            "Tumor Type": [name for name, _ in sorted_probs],
            "Probability": [f"{prob:.2%}" for _, prob in sorted_probs]
        })

    with col2:
        st.markdown("### ✅ Classification Result")
        st.write("Based on the highest predicted probability, the model provides the classification result. A confidence level is also displayed, indicating the certainty of the classification. If the confidence is low, further analysis may be needed.")
        if confidence > 0.85:
            st.success(f"**{class_names[pred_class]}** (Confidence: {confidence:.1%})")
        elif confidence > 0.7:
            st.warning(f"**{class_names[pred_class]}** (Confidence: {confidence:.1%})")
        else:
            st.error(f"**{class_names[pred_class]}** (Low Confidence: {confidence:.1%})")
        
        st.progress(int(confidence * 100))

    # ------------------------
    # 🖼️ Attention Visualization
    # ------------------------
    # Optional expandable info box for users
    with st.expander("ℹ️ What does the attention map show?", expanded=True):
        st.markdown("""
        The **attention heatmap** shows the regions of the MRI that the model is focusing on to make its prediction.
    
        It includes:
        - **Channel Attention**: Highlights which types of extracted features are most important (e.g., texture, edges).
        - **Spatial Attention**: Shows **where** those important features appear in the image.
    
        Brighter areas mean higher importance. This supports interpretability and helps clinicians trust the model’s decision-making process.
    
        > This is based on **attention mechanisms** such as those introduced in [CBAM (Woo et al., 2018)](https://arxiv.org/abs/1807.06521) and similar models in vision tasks.
        """)
    threshold_col, opacity_col = st.columns(2)
    with threshold_col:
        threshold = st.slider(
            "Attention Threshold",
            0.0, 1.0, 0.3, 0.01,
            key="attn_thresh"
        )
    with opacity_col:
        overlay_opacity = st.slider(
            "Overlay Opacity",
            0.1, 1.0, 0.5, 0.05,
            key="overlay_opacity"
        )

    fig = plt.figure(figsize=(10, 5))
    ax1, ax2 = fig.subplots(1, 2)

    # Original + Overlay
    ax1.imshow(np.array(image.resize((128, 128))), cmap='gray')
    im = ax1.imshow(attention_norm, cmap='viridis', alpha=overlay_opacity*(attention_norm >= threshold))
    fig.colorbar(im, ax=ax1, label='Attention Intensity')
    ax1.set_title("Attention Heatmap")

    # Thresholded Binary Mask
    ax2.imshow(np.array(image.resize((128, 128))), cmap='gray')
    ax2.imshow(attention_norm >= threshold, cmap='Reds', alpha=0.3)
    ax2.set_title(f"Activated Regions (≥{threshold:.2f})")

    for ax in [ax1, ax2]:
        ax.axis('off')
    st.pyplot(fig)

    # ------------------------
    # UMAP Integration
    # ------------------------

    # Load UMAP reducer
    reducer = joblib.load("umap_model.pkl")

    # Project the features from the uploaded image onto the UMAP space
    new_image_umap = reducer.transform(quantum_outputs.reshape(1, -1))

    # Load saved t-SNE outputs and labels
    outputs = np.load("all_outputs.npy")  # shape: (N_samples, 2)
    labels = np.load("all_labels.npy")    # shape: (N_samples,)

    def plot_umap(outputs, labels, new_output=None, title="UMAP of Quantum Model Outputs"):
        """
        Visualize UMAP projection with optional new point (from a new image).
        """
        # Project existing outputs
        umap_results = reducer.transform(outputs)  # 2D projection

        # Plot the new point
        fig, ax = plt.subplots(figsize=(10, 8))

        for class_id in range(4):
            idx = labels == class_id
            ax.scatter(umap_results[idx, 0], umap_results[idx, 1],
                    label=class_names[class_id], alpha=0.6, edgecolors='k', s=80)
        
        if new_output is not None:
            ax.scatter(new_output[0, 0], new_output[0, 1],
                    c='black', s=200, label='Your MRI Image', edgecolors='yellow', linewidths=2, marker='X')

        ax.set_xlabel("UMAP Component 1")
        ax.set_ylabel("UMAP Component 2")
        ax.set_title(title)
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)
    # Add user-friendly explanation
    with st.expander("🧠 UMAP Visualization", expanded=False):
        st.markdown("""
        The figure below shows a **UMAP (Uniform Manifold Approximation and Projection)** projection of the quantum model outputs. UMAP is a dimensionality reduction technique that allows us to visualize complex data in a lower-dimensional space (in this case, a 2D space).
        
        - **Colored Points**: Each point represents a sample from the training dataset. The color corresponds to the predicted class (tumor type) of that sample: **No Tumor** (Class 0), **Meningioma** (Class 1), **Glioma** (Class 2), **Pituitary Tumor** (Class 3).
        
        - **Your MRI Image**: The **black 'X'** represents the projection of the uploaded MRI image on the same 2D UMAP space. This helps us understand how the model views your image in relation to the other training samples. The closer the black 'X' is to other points of the same color, the more likely it belongs to that class (e.g., if it's close to other **Meningioma** points, the model is confident it could be a Meningioma).
        
        This visualization allows you to see how the uploaded image compares to the training data and which class (or tumor type) it most likely belongs to.
        """)

    # Plot UMAP
    plot_umap(outputs, labels, new_output=new_image_umap)

    # ------------------------
    # 🔄 Online Feedback & Online Learning
    # ------------------------




