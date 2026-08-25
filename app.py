import os
import requests

import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st

from PIL import Image
from torchvision import transforms

from model import SharedDualEfficientNetB3


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SonoInsight AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 2rem 2.2rem;
        border-radius: 20px;
        background: linear-gradient(
            135deg,
            #f7f9fc 0%,
            #edf3f8 100%
        );
        border: 1px solid #dfe7ef;
        margin-bottom: 2rem;
    }

    .hero-title {
        font-size: 2.6rem;
        font-weight: 750;
        color: #17212b;
        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #64748b;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #17212b;
        margin-top: 1.7rem;
        margin-bottom: 0.5rem;
    }

    .section-caption {
        color: #64748b;
        font-size: 0.92rem;
        margin-bottom: 1rem;
    }

    .result-card {
        padding: 1.3rem 1.5rem;
        border-radius: 16px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }

    .result-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .result-value {
        font-size: 2rem;
        font-weight: 750;
        color: #17212b;
    }

    .info-card {
        padding: 1rem 1.2rem;
        border-radius: 14px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
    }

    .info-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .info-value {
        font-size: 1rem;
        font-weight: 650;
        color: #1e293b;
        margin-top: 0.2rem;
    }

    .visual-card {
        padding: 0.9rem;
        border-radius: 16px;
        background: #ffffff;
        border: 1px solid #e2e8f0;
    }

    .visual-title {
        font-size: 1rem;
        font-weight: 700;
        color: #334155;
        margin-bottom: 0.6rem;
    }

    .disclaimer {
        padding: 1rem 1.2rem;
        border-radius: 13px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        color: #64748b;
        font-size: 0.86rem;
        line-height: 1.55;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid #e2e8f0;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 650;
    }

    [data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 1rem;
        border-radius: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

CHECKPOINT_PATH = os.path.join(
    BASE_DIR,
    "best_dual_effnet_b3.pth"
)


# ============================================================
# HUGGING FACE CHECKPOINT
# ============================================================

HF_CHECKPOINT_URL = (
    "https://huggingface.co/"
    "zara14ashraf/busbra-dual-effnet-b3/"
    "resolve/main/best_dual_effnet_b3.pth"
)


# ============================================================
# MODEL CONFIG
# ============================================================

MODEL_NAME = "SharedDualEfficientNetB3"

IMAGE_SIZE = 300

THRESHOLD = 0.52

CROP_MARGIN = 0.25

MEAN = [
    0.485,
    0.456,
    0.406
]

STD = [
    0.229,
    0.224,
    0.225
]


# ============================================================
# SAMPLE CASES
# ============================================================

SAMPLES = {

    "Case 01 · Benign": {
        "id": "bus_0002-l",
        "image": "sample_1_bus_0002-l.png",
        "mask": "sample_1_bus_0002-l_MASK.png",
        "bbox": [134, 142, 88, 50],
        "label": "Benign",
        "histology": "Fibroadenoma",
    },

    "Case 02 · Benign": {
        "id": "bus_0002-r",
        "image": "sample_2_bus_0002-r.png",
        "mask": "sample_2_bus_0002-r_MASK.png",
        "bbox": [113, 143, 68, 47],
        "label": "Benign",
        "histology": "Fibroadenoma",
    },

    "Case 03 · Malignant": {
        "id": "bus_0001-l",
        "image": "sample_3_bus_0001-l.png",
        "mask": "sample_3_bus_0001-l_MASK.png",
        "bbox": [91, 24, 103, 79],
        "label": "Malignant",
        "histology": "Invasive ductal carcinoma",
    },

    "Case 04 · Malignant": {
        "id": "bus_0001-r",
        "image": "sample_4_bus_0001-r.png",
        "mask": "sample_4_bus_0001-r_MASK.png",
        "bbox": [102, 24, 82, 79],
        "label": "Malignant",
        "histology": "Invasive ductal carcinoma",
    },
}


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# DOWNLOAD CHECKPOINT
# ============================================================

def download_checkpoint():

    if os.path.exists(CHECKPOINT_PATH):
        return

    response = requests.get(
        HF_CHECKPOINT_URL,
        stream=True,
        timeout=300
    )

    response.raise_for_status()

    with open(
        CHECKPOINT_PATH,
        "wb"
    ) as file:

        for chunk in response.iter_content(
            chunk_size=1024 * 1024
        ):

            if chunk:
                file.write(chunk)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    download_checkpoint()

    model = SharedDualEfficientNetB3(
        num_classes=2
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False
    )

    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        else:
            state_dict = checkpoint

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model = model.to(device)

    model.eval()

    return model


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=MEAN,
        std=STD
    ),
])


# ============================================================
# LESION CROP FROM BBOX
# ============================================================

def make_lesion_crop(
    image,
    bbox,
    margin=CROP_MARGIN
):

    x, y, width, height = [
        int(v)
        for v in bbox
    ]

    image_width, image_height = image.size

    pad_x = int(width * margin)
    pad_y = int(height * margin)

    x1 = max(
        0,
        x - pad_x
    )

    y1 = max(
        0,
        y - pad_y
    )

    x2 = min(
        image_width,
        x + width + pad_x
    )

    y2 = min(
        image_height,
        y + height + pad_y
    )

    return image.crop(
        (x1, y1, x2, y2)
    )


# ============================================================
# PREDICTION
# ============================================================

def predict_with_crop(
    model,
    image,
    crop_image
):

    full_tensor = transform(
        image
    ).unsqueeze(
        0
    ).to(device)

    crop_tensor = transform(
        crop_image
    ).unsqueeze(
        0
    ).to(device)

    with torch.no_grad():

        logits = model(
            full_tensor,
            crop_tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

    benign = probabilities[0].item()

    malignant = probabilities[1].item()

    prediction = (
        "Malignant"
        if malignant >= THRESHOLD
        else "Benign"
    )

    return (
        prediction,
        benign,
        malignant,
        full_tensor,
        crop_tensor,
    )


# ============================================================
# GRAD-CAM
# ============================================================

class GradCAM:

    def __init__(
        self,
        model,
        target_layer
    ):

        self.model = model

        self.activations = None
        self.gradients = None

        self.forward_hook = (
            target_layer.register_forward_hook(
                self._save_activation
            )
        )

        self.backward_hook = (
            target_layer.register_full_backward_hook(
                self._save_gradient
            )
        )

    def _save_activation(
        self,
        module,
        inputs,
        output
    ):

        self.activations = output

    def _save_gradient(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = grad_output[0]

    def generate(
        self,
        full_tensor,
        crop_tensor,
        target_class
    ):

        self.model.zero_grad(
            set_to_none=True
        )

        logits = self.model(
            full_tensor,
            crop_tensor
        )

        score = logits[
            0,
            target_class
        ]

        score.backward()

        activations = self.activations

        gradients = self.gradients

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (
            weights * activations
        ).sum(
            dim=1,
            keepdim=True
        )

        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=(
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            mode="bilinear",
            align_corners=False
        )

        cam = cam[
            0,
            0
        ]

        cam = (
            cam
            .detach()
            .cpu()
            .numpy()
        )

        cam -= cam.min()

        maximum = cam.max()

        if maximum > 0:
            cam /= maximum

        return cam

    def remove_hooks(self):

        self.forward_hook.remove()
        self.backward_hook.remove()


# ============================================================
# GENERATE GRAD-CAM
# ============================================================

def generate_gradcam(
    model,
    full_tensor,
    crop_tensor,
    predicted_class
):

    target_layer = (
        model.full_branch.blocks[-1]
    )

    gradcam = GradCAM(
        model,
        target_layer
    )

    try:

        cam = gradcam.generate(
            full_tensor,
            crop_tensor,
            predicted_class
        )

    finally:

        gradcam.remove_hooks()

    return cam


# ============================================================
# GRAD-CAM OVERLAY
# ============================================================

def create_gradcam_overlay(
    image,
    cam
):

    image_resized = image.resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    )

    image_array = np.array(
        image_resized
    ).astype(
        np.float32
    ) / 255.0

    cam_uint8 = (
        cam * 255
    ).astype(
        np.uint8
    )

    heatmap = np.zeros(
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
            3
        ),
        dtype=np.float32
    )

    heatmap[:, :, 0] = cam_uint8

    heatmap[:, :, 1] = (
        255 - cam_uint8
    ) * 0.5

    heatmap[:, :, 2] = (
        255 - cam_uint8
    )

    heatmap /= 255.0

    overlay = (
        0.55 * image_array
        + 0.45 * heatmap
    )

    overlay = np.clip(
        overlay,
        0,
        1
    )

    return Image.fromarray(
        (
            overlay * 255
        ).astype(
            np.uint8
        )
    )


# ============================================================
# AUTOMATIC CROP FROM GRAD-CAM
# ============================================================

def bbox_from_cam(
    cam,
    original_size,
    threshold_ratio=0.55,
    padding_ratio=0.20
):

    height, width = cam.shape

    threshold = (
        cam.max()
        * threshold_ratio
    )

    mask = cam >= threshold

    if not mask.any():

        return [
            0,
            0,
            original_size[0],
            original_size[1]
        ]

    ys, xs = np.where(mask)

    x1 = xs.min()
    x2 = xs.max()

    y1 = ys.min()
    y2 = ys.max()

    cam_width = x2 - x1 + 1
    cam_height = y2 - y1 + 1

    pad_x = int(
        cam_width
        * padding_ratio
    )

    pad_y = int(
        cam_height
        * padding_ratio
    )

    x1 -= pad_x
    y1 -= pad_y
    x2 += pad_x
    y2 += pad_y

    original_width, original_height = (
        original_size
    )

    scale_x = (
        original_width
        / IMAGE_SIZE
    )

    scale_y = (
        original_height
        / IMAGE_SIZE
    )

    x1 = int(
        max(
            0,
            x1 * scale_x
        )
    )

    y1 = int(
        max(
            0,
            y1 * scale_y
        )
    )

    x2 = int(
        min(
            original_width,
            x2 * scale_x
        )
    )

    y2 = int(
        min(
            original_height,
            y2 * scale_y
        )
    )

    if x2 <= x1 or y2 <= y1:

        return [
            0,
            0,
            original_width,
            original_height
        ]

    return [
        x1,
        y1,
        x2 - x1,
        y2 - y1
    ]


# ============================================================
# AUTOMATIC LESION CROP
# ============================================================

def generate_attention_crop(
    model,
    image
):

    # Initial pass:
    # full image is temporarily used for both branches.

    initial_full = transform(
        image
    ).unsqueeze(
        0
    ).to(device)

    initial_crop = initial_full.clone()

    with torch.no_grad():

        logits = model(
            initial_full,
            initial_crop
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

    initial_class = int(
        torch.argmax(
            probabilities
        ).item()
    )

    # Generate attention map
    cam = generate_gradcam(
        model,
        initial_full,
        initial_crop,
        initial_class
    )

    # Convert attention into an ROI
    bbox = bbox_from_cam(
        cam,
        image.size
    )

    crop_image = make_lesion_crop(
        image,
        bbox,
        margin=0.0
    )

    return (
        crop_image,
        bbox,
        cam
    )


# ============================================================
# VISUAL CARD
# ============================================================

def show_visual(
    title,
    image,
    caption
):

    st.markdown(
        f"""
        <div class="visual-card">

            <div class="visual-title">
                {title}
            </div>

        """,
        unsafe_allow_html=True
    )

    st.image(
        image,
        use_container_width=True
    )

    st.caption(
        caption
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            SonoInsight AI
        </div>

        <div class="hero-subtitle">
            AI-Assisted Breast Ultrasound Analysis
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## Analysis"
    )

    mode = st.radio(
        "Choose analysis mode",
        [
            "Explore Sample Cases",
            "Analyze Your Ultrasound"
        ]
    )

    st.divider()

    st.markdown(
        "### About SonoInsight AI"
    )

    st.caption(
        "A research prototype demonstrating "
        "dual-view deep learning for breast "
        "ultrasound classification and "
        "visual explainability."
    )


# ============================================================
# LOAD MODEL
# ============================================================

try:

    with st.spinner(
        "Preparing AI model..."
    ):

        model = load_model()

except Exception as error:

    st.error(
        "Unable to load the trained model."
    )

    st.exception(error)

    st.stop()


# ============================================================
# SAMPLE CASE MODE
# ============================================================

if mode == "Explore Sample Cases":

    st.markdown(
        '<div class="section-title">Explore Sample Cases</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-caption">
            Explore representative ultrasound cases
            with model predictions and explainable AI.
        </div>
        """,
        unsafe_allow_html=True
    )

    selected = st.selectbox(
        "Select a case",
        list(SAMPLES.keys())
    )

    sample = SAMPLES[selected]

    image_path = os.path.join(
        BASE_DIR,
        sample["image"]
    )

    mask_path = os.path.join(
        BASE_DIR,
        sample["mask"]
    )

    if not os.path.exists(image_path):

        st.error(
            "Sample image not found."
        )

        st.stop()

    image = Image.open(
        image_path
    ).convert("RGB")

    mask = None

    if os.path.exists(mask_path):

        mask = Image.open(
            mask_path
        ).convert("L")

    bbox = sample["bbox"]

    crop_image = make_lesion_crop(
        image,
        bbox
    )

    try:

        (
            prediction,
            benign,
            malignant,
            full_tensor,
            crop_tensor
        ) = predict_with_crop(
            model,
            image,
            crop_image
        )

        predicted_class = (
            1
            if prediction == "Malignant"
            else 0
        )

        cam = generate_gradcam(
            model,
            full_tensor,
            crop_tensor,
            predicted_class
        )

        gradcam = create_gradcam_overlay(
            image,
            cam
        )

    except Exception as error:

        st.error(
            "Sample analysis failed."
        )

        st.exception(error)

        st.stop()

    # --------------------------------------------------------
    # CASE INFORMATION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Case Overview</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-label">
                    Case
                </div>

                <div class="info-value">
                    {sample["id"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-label">
                    Pathology
                </div>

                <div class="info-value">
                    {sample["label"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-label">
                    Histology
                </div>

                <div class="info-value">
                    {sample["histology"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # AI ASSESSMENT
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">AI Assessment</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="result-card">

            <div class="result-label">
                Model prediction
            </div>

            <div class="result-value">
                {prediction}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    p1, p2 = st.columns(2)

    with p1:

        st.metric(
            "Benign probability",
            f"{benign * 100:.1f}%"
        )

    with p2:

        st.metric(
            "Malignant probability",
            f"{malignant * 100:.1f}%"
        )

    # --------------------------------------------------------
    # VISUAL EXPLANATION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Explainable Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-caption">
            Compare the annotated lesion with the regions
            receiving the model's attention.
        </div>
        """,
        unsafe_allow_html=True
    )

    if mask is not None:

        from_mask = create_mask_overlay(
            image,
            mask
        )

    else:

        from_mask = image

    v1, v2, v3 = st.columns(3)

    with v1:

        show_visual(
            "Ultrasound & Ground Truth",
            from_mask,
            "Ground-truth lesion annotation."
        )

    with v2:

        show_visual(
            "Grad-CAM Attention",
            gradcam,
            "Regions contributing to the prediction."
        )

    with v3:

        show_visual(
            "Lesion-Focused View",
            crop_image,
            "Dataset BBOX-based second model view."
        )


# ============================================================
# UPLOAD MODE
# ============================================================

else:

    st.markdown(
        '<div class="section-title">Analyze Your Ultrasound</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-caption">
            Upload a breast ultrasound image and let
            SonoInsight AI automatically generate a
            lesion-focused model view.
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload ultrasound image",
        type=[
            "png",
            "jpg",
            "jpeg"
        ]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            caption="Uploaded ultrasound",
            width=500
        )

        if st.button(
            "Analyze Ultrasound",
            type="primary",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "Analyzing ultrasound..."
                ):

                    # ----------------------------------------
                    # AUTOMATIC LESION LOCALIZATION
                    # ----------------------------------------

                    (
                        auto_crop,
                        auto_bbox,
                        initial_cam
                    ) = generate_attention_crop(
                        model,
                        image
                    )

                    # ----------------------------------------
                    # FINAL DUAL-VIEW PREDICTION
                    # ----------------------------------------

                    (
                        prediction,
                        benign,
                        malignant,
                        full_tensor,
                        crop_tensor
                    ) = predict_with_crop(
                        model,
                        image,
                        auto_crop
                    )

                    predicted_class = (
                        1
                        if prediction == "Malignant"
                        else 0
                    )

                    # ----------------------------------------
                    # FINAL GRAD-CAM
                    # ----------------------------------------

                    final_cam = generate_gradcam(
                        model,
                        full_tensor,
                        crop_tensor,
                        predicted_class
                    )

                    gradcam = create_gradcam_overlay(
                        image,
                        final_cam
                    )

                # ------------------------------------------------
                # RESULT
                # ------------------------------------------------

                st.markdown(
                    '<div class="section-title">AI Assessment</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"""
                    <div class="result-card">

                        <div class="result-label">
                            Model prediction
                        </div>

                        <div class="result-value">
                            {prediction}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                p1, p2 = st.columns(2)

                with p1:

                    st.metric(
                        "Benign probability",
                        f"{benign * 100:.1f}%"
                    )

                with p2:

                    st.metric(
                        "Malignant probability",
                        f"{malignant * 100:.1f}%"
                    )

                # ------------------------------------------------
                # EXPLAINABILITY
                # ------------------------------------------------

                st.markdown(
                    '<div class="section-title">Explainable Analysis</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    """
                    <div class="section-caption">
                        The lesion-focused view is automatically
                        generated from the model's attention.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                v1, v2, v3 = st.columns(3)

                with v1:

                    show_visual(
                        "Uploaded Ultrasound",
                        image,
                        "Original uploaded ultrasound."
                    )

                with v2:

                    show_visual(
                        "Grad-CAM Attention",
                        gradcam,
                        "Regions contributing to the prediction."
                    )

                with v3:

                    show_visual(
                        "AI-Generated Lesion View",
                        auto_crop,
                        "Automatically generated attention-guided crop."
                    )

            except Exception as error:

                st.error(
                    "Prediction failed."
                )

                st.exception(error)


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

with st.expander(
    "Model & Methodology"
):

    c1, c2 = st.columns(2)

    with c1:

        st.write(
            "**Architecture:** Dual EfficientNet-B3"
        )

        st.write(
            "**Task:** Benign vs Malignant"
        )

        st.write(
            f"**Input resolution:** "
            f"{IMAGE_SIZE} × {IMAGE_SIZE}"
        )

    with c2:

        st.write(
            "**Explainability:** Grad-CAM"
        )

        st.write(
            "**Sample localization:** Dataset BBOX"
        )

        st.write(
            "**Upload localization:** Attention-guided crop"
        )


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown(
    """
    <div class="disclaimer">

    <strong>Research & Demonstration Prototype</strong><br>

    This is a demonstration prototype for AI-assisted
    breast ultrasound analysis, trained on the BUS-BRA dataset.
    It is intended for research and educational purposes only
    and is not a clinical diagnostic tool. Model performance
    may improve with larger, more diverse datasets and
    further training.

    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<br>",
    unsafe_allow_html=True
)

st.caption(
    "SonoInsight AI · Research Prototype"
)
