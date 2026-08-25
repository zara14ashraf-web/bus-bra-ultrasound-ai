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
    page_title="AI-Assisted Breast Ultrasound Analysis",
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

    /* ========================================================
       GLOBAL
       ======================================================== */

    .stApp {
        background: #ffffff;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4 {
        color: #163f4d;
    }

    /* ========================================================
       HERO
       ======================================================== */

    .hero {
        text-align: center;
        padding: 1.8rem 1rem 1.2rem 1rem;
    }

    .hero-badge {
        display: inline-block;
        padding: 5px 13px;
        border: 1px solid #d7e5e9;
        border-radius: 999px;
        background: #f5fafb;
        color: #397080;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.35px;
        margin-bottom: 0.8rem;
    }

    .hero-title {
        font-size: 2.55rem;
        line-height: 1.15;
        font-weight: 750;
        color: #123d4b;
        letter-spacing: -1.2px;
        margin-bottom: 0.45rem;
    }

    .hero-subtitle {
        font-size: 1.08rem;
        color: #58727b;
        margin-bottom: 0.75rem;
    }

    .hero-description {
        max-width: 760px;
        margin: auto;
        color: #687b83;
        font-size: 0.91rem;
        line-height: 1.65;
    }

    /* ========================================================
       DISCLAIMER
       ======================================================== */

    .disclaimer {
        background: #f8fafc;
        border: 1px solid #dbe5ea;
        border-left: 4px solid #5c8995;
        border-radius: 10px;
        padding: 13px 17px;
        margin: 0.8rem 0 1.8rem 0;
        color: #52636b;
        font-size: 0.82rem;
        line-height: 1.55;
    }

    .disclaimer-title {
        color: #183f4d;
        font-weight: 700;
        margin-bottom: 3px;
    }

    /* ========================================================
       SECTION
       ======================================================== */

    .section-header {
        margin-top: 1.8rem;
        margin-bottom: 0.85rem;
    }

    .section-title {
        color: #183f4d;
        font-size: 1.38rem;
        font-weight: 720;
        margin-bottom: 0.18rem;
    }

    .section-subtitle {
        color: #74838a;
        font-size: 0.84rem;
        line-height: 1.5;
    }

    /* ========================================================
       MODEL CARDS
       ======================================================== */

    .model-card {
        background: #f8fafb;
        border: 1px solid #e0e8eb;
        border-radius: 11px;
        padding: 15px 16px;
        min-height: 92px;
    }

    .model-label {
        color: #7b8b92;
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.65px;
        margin-bottom: 5px;
    }

    .model-value {
        color: #183f4d;
        font-size: 0.92rem;
        font-weight: 680;
        line-height: 1.3;
    }

    /* ========================================================
       WHY SECTION
       ======================================================== */

    .why-box {
        background: #f8fafb;
        border: 1px solid #e0e8eb;
        border-radius: 12px;
        padding: 19px 21px;
        color: #53656d;
        font-size: 0.88rem;
        line-height: 1.72;
    }

    .why-highlight {
        color: #214f5d;
        font-weight: 650;
    }

    /* ========================================================
       WORKSPACE
       ======================================================== */

    .workspace {
        background: #f9fbfc;
        border: 1px solid #dfe8eb;
        border-radius: 14px;
        padding: 20px;
        margin-top: 0.8rem;
    }

    .workspace-title {
        color: #183f4d;
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 3px;
    }

    .workspace-text {
        color: #75848a;
        font-size: 0.8rem;
        line-height: 1.5;
    }

    /* ========================================================
       RESULT
       ======================================================== */

    .prediction-panel {
        background: #ffffff;
        border: 1px solid #dfe8eb;
        border-radius: 13px;
        padding: 19px 20px;
        margin-top: 0.7rem;
    }

    .prediction-label {
        color: #7b898f;
        text-transform: uppercase;
        letter-spacing: 0.7px;
        font-size: 0.67rem;
        font-weight: 650;
    }

    .prediction-value {
        font-size: 2rem;
        line-height: 1.2;
        font-weight: 760;
        margin-top: 5px;
    }

    .prediction-note {
        color: #7b898f;
        font-size: 0.75rem;
        margin-top: 4px;
    }

    /* ========================================================
       PROBABILITY
       ======================================================== */

    .prob-card {
        background: #ffffff;
        border: 1px solid #dfe8eb;
        border-radius: 11px;
        padding: 14px 16px;
        margin-bottom: 0.65rem;
    }

    .prob-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 7px;
    }

    .prob-name {
        color: #52656d;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .prob-value {
        color: #183f4d;
        font-size: 0.86rem;
        font-weight: 720;
    }

    .prob-track {
        width: 100%;
        height: 7px;
        background: #e8eef0;
        border-radius: 20px;
        overflow: hidden;
    }

    .prob-fill {
        height: 100%;
        border-radius: 20px;
    }

    .benign-fill {
        background: #4d9277;
    }

    .malignant-fill {
        background: #b75c5c;
    }

    /* ========================================================
       VISUAL EXPLANATION
       ======================================================== */

    .explain-card {
        background: #f9fbfc;
        border: 1px solid #dfe8eb;
        border-radius: 12px;
        padding: 14px;
        min-height: 70px;
    }

    .explain-title {
        color: #183f4d;
        font-size: 0.86rem;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .explain-text {
        color: #74838a;
        font-size: 0.76rem;
        line-height: 1.45;
    }

    /* ========================================================
       SAMPLE CASE
       ======================================================== */

    .sample-card {
        background: #ffffff;
        border: 1px solid #dfe8eb;
        border-radius: 12px;
        padding: 10px;
    }

    .sample-name {
        color: #183f4d;
        font-size: 0.82rem;
        font-weight: 680;
        margin-top: 6px;
    }

    .sample-id {
        color: #849198;
        font-size: 0.69rem;
        margin-top: 2px;
    }

    /* ========================================================
       CASE META
       ======================================================== */

    .case-meta {
        background: #f8fafb;
        border: 1px solid #e0e8eb;
        border-radius: 10px;
        padding: 12px 14px;
    }

    .case-meta-label {
        color: #7c8a91;
        font-size: 0.66rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .case-meta-value {
        color: #183f4d;
        font-size: 0.84rem;
        font-weight: 650;
        margin-top: 3px;
    }

    /* ========================================================
       UPLOAD
       ======================================================== */

    .upload-note {
        background: #f8fafb;
        border: 1px dashed #cfdde1;
        border-radius: 11px;
        padding: 14px;
        color: #718087;
        font-size: 0.78rem;
        text-align: center;
        margin-top: 0.5rem;
    }

    /* ========================================================
       SIDEBAR
       ======================================================== */

    [data-testid="stSidebar"] {
        background: #f7fafb;
        border-right: 1px solid #e1e9ec;
    }

    .sidebar-brand {
        color: #163f4d;
        font-size: 1.05rem;
        font-weight: 750;
        line-height: 1.3;
    }

    .sidebar-subtitle {
        color: #77868d;
        font-size: 0.74rem;
        line-height: 1.5;
        margin-top: 4px;
    }

    .sidebar-heading {
        color: #234c59;
        font-size: 0.75rem;
        font-weight: 720;
        text-transform: uppercase;
        letter-spacing: 0.45px;
        margin-top: 1rem;
        margin-bottom: 4px;
    }

    .sidebar-text {
        color: #687a81;
        font-size: 0.76rem;
        line-height: 1.55;
    }

    .sidebar-author {
        color: #234c59;
        font-size: 0.82rem;
        font-weight: 700;
    }

    /* ========================================================
       FOOTER
       ======================================================== */

    .footer {
        text-align: center;
        padding-top: 1.8rem;
        margin-top: 2rem;
        border-top: 1px solid #edf1f2;
        color: #8b999f;
        font-size: 0.72rem;
        line-height: 1.6;
    }

    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }

    /* ========================================================
       FILE UPLOADER
       ======================================================== */

    [data-testid="stFileUploader"] {
        border-radius: 10px;
    }

    /* ========================================================
       MOBILE
       ======================================================== */

    @media (max-width: 768px) {

        .hero-title {
            font-size: 2rem;
        }

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

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

    "Case 01 — Benign": {
        "id": "bus_0002-l",
        "image": "sample_1_bus_0002-l.png",
        "mask": "sample_1_bus_0002-l_MASK.png",
        "bbox": [134, 142, 88, 50],
        "label": "Benign",
        "histology": "Fibroadenoma",
    },

    "Case 02 — Benign": {
        "id": "bus_0002-r",
        "image": "sample_2_bus_0002-r.png",
        "mask": "sample_2_bus_0002-r_MASK.png",
        "bbox": [113, 143, 68, 47],
        "label": "Benign",
        "histology": "Fibroadenoma",
    },

    "Case 03 — Malignant": {
        "id": "bus_0001-l",
        "image": "sample_3_bus_0001-l.png",
        "mask": "sample_3_bus_0001-l_MASK.png",
        "bbox": [91, 24, 103, 79],
        "label": "Malignant",
        "histology": "Invasive ductal carcinoma",
    },

    "Case 04 — Malignant": {
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
# DOWNLOAD MODEL
# ============================================================

def download_checkpoint():

    if os.path.exists(CHECKPOINT_PATH):
        return

    with st.spinner(
        "Preparing the trained model..."
    ):

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
# IMAGE TRANSFORM
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
# LESION CROP
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
# MASK OVERLAY
# ============================================================

def create_mask_overlay(
    image,
    mask
):

    image = image.convert("RGB")
    mask = mask.convert("L")

    if mask.size != image.size:

        mask = mask.resize(
            image.size
        )

    image_array = np.array(
        image
    ).astype(
        np.float32
    )

    mask_array = np.array(
        mask
    )

    mask_binary = (
        mask_array > 0
    )

    if not mask_binary.any():
        return image

    overlay = image_array.copy()

    overlay[
        mask_binary,
        0
    ] = 255

    overlay[
        mask_binary,
        1
    ] *= 0.35

    overlay[
        mask_binary,
        2
    ] *= 0.35

    result = (
        0.65 * image_array
        + 0.35 * overlay
    )

    result = np.clip(
        result,
        0,
        255
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        result
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
            cam.detach()
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
# PREDICTION
# ============================================================

def predict(
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

    benign_probability = (
        probabilities[0].item()
    )

    malignant_probability = (
        probabilities[1].item()
    )

    prediction = (
        "Malignant"
        if malignant_probability >= THRESHOLD
        else "Benign"
    )

    return (
        prediction,
        benign_probability,
        malignant_probability,
        full_tensor,
        crop_tensor,
    )


# ============================================================
# BBOX FROM CAM
# ============================================================

def bbox_from_cam(
    cam,
    original_size,
    threshold_ratio=0.55,
    padding_ratio=0.20
):

    image_width, image_height = (
        original_size
    )

    threshold = (
        cam.max()
        * threshold_ratio
    )

    active = cam >= threshold

    if not active.any():

        return [
            0,
            0,
            image_width,
            image_height
        ]

    ys, xs = np.where(
        active
    )

    x1 = xs.min()
    x2 = xs.max()

    y1 = ys.min()
    y2 = ys.max()

    roi_width = (
        x2 - x1 + 1
    )

    roi_height = (
        y2 - y1 + 1
    )

    pad_x = int(
        roi_width
        * padding_ratio
    )

    pad_y = int(
        roi_height
        * padding_ratio
    )

    x1 -= pad_x
    y1 -= pad_y
    x2 += pad_x
    y2 += pad_y

    scale_x = (
        image_width
        / IMAGE_SIZE
    )

    scale_y = (
        image_height
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
            image_width,
            x2 * scale_x
        )
    )

    y2 = int(
        min(
            image_height,
            y2 * scale_y
        )
    )

    if (
        x2 <= x1
        or y2 <= y1
    ):

        return [
            0,
            0,
            image_width,
            image_height
        ]

    return [
        x1,
        y1,
        x2 - x1,
        y2 - y1
    ]


# ============================================================
# AUTOMATIC UPLOAD CROP
# ============================================================

def generate_automatic_crop(
    model,
    image
):

    full_tensor = transform(
        image
    ).unsqueeze(
        0
    ).to(device)

    with torch.no_grad():

        logits = model(
            full_tensor,
            full_tensor
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

    cam = generate_gradcam(
        model,
        full_tensor,
        full_tensor,
        initial_class
    )

    bbox = bbox_from_cam(
        cam,
        image.size
    )

    crop = make_lesion_crop(
        image,
        bbox,
        margin=0.0
    )

    return crop


# ============================================================
# UI HELPERS
# ============================================================

def render_probability(
    name,
    value,
    css_class
):

    percentage = value * 100

    st.markdown(
        f"""
        <div class="prob-card">

            <div class="prob-header">

                <div class="prob-name">
                    {name}
                </div>

                <div class="prob-value">
                    {percentage:.1f}%
                </div>

            </div>

            <div class="prob-track">

                <div
                    class="prob-fill {css_class}"
                    style="width:{percentage:.1f}%"
                ></div>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


def render_prediction(
    prediction,
    benign_probability,
    malignant_probability
):

    if prediction == "Malignant":

        result_color = "#a94d4d"

        interpretation = (
            "The model assigns the higher probability "
            "to the malignant class."
        )

    else:

        result_color = "#438568"

        interpretation = (
            "The model assigns the higher probability "
            "to the benign class."
        )

    left, right = st.columns(
        [0.9, 1.1],
        gap="large"
    )

    with left:

        st.markdown(
            f"""
            <div class="prediction-panel">

                <div class="prediction-label">
                    AI-assisted prediction
                </div>

                <div
                    class="prediction-value"
                    style="color:{result_color};"
                >
                    {prediction}
                </div>

                <div class="prediction-note">
                    {interpretation}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with right:

        render_probability(
            "Benign",
            benign_probability,
            "benign-fill"
        )

        render_probability(
            "Malignant",
            malignant_probability,
            "malignant-fill"
        )


def render_explanation_section(
    image,
    gradcam_image,
    crop_image,
    mask=None
):

    st.markdown(
        """
        <div class="section-header">

            <div class="section-title">
                Visual Explanation
            </div>

            <div class="section-subtitle">
                Grad-CAM provides a visual indication of regions
                that contributed to the model prediction.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    if mask is not None:

        mask_overlay = create_mask_overlay(
            image,
            mask
        )

        v1, v2, v3 = st.columns(
            3,
            gap="medium"
        )

        with v1:

            st.image(
                image,
                caption="Original Ultrasound",
                use_container_width=True
            )

        with v2:

            st.image(
                mask_overlay,
                caption="Reference Lesion Mask",
                use_container_width=True
            )

        with v3:

            st.image(
                gradcam_image,
                caption="Grad-CAM Attention",
                use_container_width=True
            )

        st.markdown(
            """
            <div class="explain-card">

                <div class="explain-title">
                    Reference annotation available
                </div>

                <div class="explain-text">
                    For representative BUS-BRA cases, the reference
                    lesion mask can be viewed alongside the model's
                    Grad-CAM attention map.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        v1, v2, v3 = st.columns(
            3,
            gap="medium"
        )

        with v1:

            st.image(
                image,
                caption="Uploaded Ultrasound",
                use_container_width=True
            )

        with v2:

            st.image(
                crop_image,
                caption="AI-Generated Lesion View",
                use_container_width=True
            )

        with v3:

            st.image(
                gradcam_image,
                caption="Grad-CAM Attention",
                use_container_width=True
            )

        st.markdown(
            """
            <div class="explain-card">

                <div class="explain-title">
                    No ground-truth annotation
                </div>

                <div class="explain-text">
                    Uploaded images do not contain reference lesion
                    masks. The highlighted region represents model
                    attention and should not be interpreted as a
                    definitive lesion segmentation.
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
        """
        <div class="sidebar-brand">
            AI-Assisted Breast Ultrasound Analysis
        </div>

        <div class="sidebar-subtitle">
            Breast ultrasound research prototype using
            dual-view deep learning and explainable AI.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        """
        <div class="sidebar-heading">
            About the Project
        </div>

        <div class="sidebar-text">
            This project explores whether a dual-view deep learning
            model can classify breast ultrasound lesions while
            providing a visual explanation of model attention.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-heading">
            Model Classes
        </div>

        <div class="sidebar-text">
            <strong>Benign</strong><br>
            <strong>Malignant</strong>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-heading">
            Dual-View Architecture
        </div>

        <div class="sidebar-text">
            The model combines information from the full ultrasound
            image with a lesion-focused image representation.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-heading">
            Explainability
        </div>

        <div class="sidebar-text">
            Grad-CAM highlights image regions associated with
            the selected prediction.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-heading">
            Workflow
        </div>

        <div class="sidebar-text">
            01 — Select a sample or upload an image<br>
            02 — Generate AI prediction<br>
            03 — Review probabilities<br>
            04 — Explore visual explanation
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        """
        <div class="sidebar-heading">
            Developed By
        </div>

        <div class="sidebar-author">
            Zara Ashraf
        </div>

        <div class="sidebar-text">
            BS Medical Imaging Technology
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-heading">
            Important
        </div>

        <div class="sidebar-text">
            Research and educational use only.
            This system is not intended for clinical diagnosis
            or treatment decisions.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-badge">
            BREAST ULTRASOUND • DEEP LEARNING • EXPLAINABLE AI
        </div>

        <div class="hero-title">
            AI-Assisted Breast Ultrasound Analysis
        </div>

        <div class="hero-subtitle">
            Dual-view deep learning for benign and malignant
            lesion classification
        </div>

        <div class="hero-description">
            A research prototype exploring how artificial intelligence
            can complement medical imaging workflows by combining
            whole-image and lesion-focused ultrasound information,
            with visual interpretation through Grad-CAM.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown(
    """
    <div class="disclaimer">

        <div class="disclaimer-title">
            Research & Educational Use Only
        </div>

        This application is a research prototype developed for
        breast ultrasound image analysis. It is not a medical device
        and should not be used to diagnose disease, exclude disease,
        or guide treatment. AI predictions should never replace
        assessment by a qualified healthcare professional.

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL AT A GLANCE
# ============================================================

st.markdown(
    """
    <div class="section-header">

        <div class="section-title">
            Model at a Glance
        </div>

        <div class="section-subtitle">
            A concise overview of the system used for this research prototype.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(
    4,
    gap="medium"
)

with c1:

    st.markdown(
        """
        <div class="model-card">

            <div class="model-label">
                Architecture
            </div>

            <div class="model-value">
                Shared Dual<br>
                EfficientNet-B3
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

with c2:

    st.markdown(
        """
        <div class="model-card">

            <div class="model-label">
                Classification
            </div>

            <div class="model-value">
                Benign<br>
                vs Malignant
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

with c3:

    st.markdown(
        f"""
        <div class="model-card">

            <div class="model-label">
                Input Resolution
            </div>

            <div class="model-value">
                {IMAGE_SIZE} × {IMAGE_SIZE}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

with c4:

    st.markdown(
        """
        <div class="model-card">

            <div class="model-label">
                Explainability
            </div>

            <div class="model-value">
                Grad-CAM
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# WHY THIS PROJECT
# ============================================================

st.markdown(
    """
    <div class="section-header">

        <div class="section-title">
            Why I Developed This Project
        </div>

    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="why-box">

        As a <span class="why-highlight">Medical Imaging Technologist</span>,
        I developed this project to explore how artificial intelligence
        can complement medical imaging workflows and make breast
        ultrasound analysis more transparent.

        <br><br>

        Breast ultrasound images can demonstrate considerable variation
        in lesion appearance, making consistent image-based assessment
        challenging. This project investigates whether a dual-view
        deep learning approach can learn from both the overall ultrasound
        image and a focused lesion representation.

        <br><br>

        <span class="why-highlight">
        The goal is not to replace the radiologist or medical imaging
        professional, but to explore how AI can provide an additional,
        explainable perspective during image analysis.
        </span>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = load_model()

except Exception as error:

    st.error(
        "Unable to load the trained AI model."
    )

    st.exception(error)

    st.stop()


# ============================================================
# ANALYSIS SECTION
# ============================================================

st.markdown(
    """
    <div class="section-header">

        <div class="section-title">
            Explore the Analysis
        </div>

        <div class="section-subtitle">
            Start with a representative BUS-BRA case or upload
            your own ultrasound image for a research demonstration.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


analysis_mode = st.radio(
    "Choose analysis mode",
    [
        "Sample Cases",
        "Upload Ultrasound"
    ],
    horizontal=True,
    label_visibility="collapsed"
)


# ============================================================
# SAMPLE CASES
# ============================================================

if analysis_mode == "Sample Cases":

    st.markdown(
        """
        <div class="workspace">

            <div class="workspace-title">
                Representative BUS-BRA Cases
            </div>

            <div class="workspace-text">
                Explore reference cases with known lesion annotations
                and compare the model's prediction with Grad-CAM attention.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    sample_names = list(
        SAMPLES.keys()
    )

    if "selected_sample" not in st.session_state:

        st.session_state.selected_sample = (
            sample_names[0]
        )

    sample_cols = st.columns(
        4,
        gap="medium"
    )

    for index, sample_name in enumerate(
        sample_names
    ):

        sample = SAMPLES[
            sample_name
        ]

        image_path = os.path.join(
            BASE_DIR,
            sample["image"]
        )

        with sample_cols[index]:

            if os.path.exists(
                image_path
            ):

                preview = Image.open(
                    image_path
                ).convert("RGB")

                st.image(
                    preview,
                    use_container_width=True
                )

            st.markdown(
                f"""
                <div class="sample-name">
                    {sample_name}
                </div>

                <div class="sample-id">
                    {sample["id"]}
                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button(
                "Explore case",
                key=f"sample_{index}",
                use_container_width=True
            ):

                st.session_state.selected_sample = (
                    sample_name
                )

    selected_sample = (
        st.session_state.selected_sample
    )

    sample = SAMPLES[
        selected_sample
    ]

    image_path = os.path.join(
        BASE_DIR,
        sample["image"]
    )

    mask_path = os.path.join(
        BASE_DIR,
        sample["mask"]
    )

    if not os.path.exists(
        image_path
    ):

        st.error(
            f"Sample image not found: {sample['image']}"
        )

        st.stop()

    image = Image.open(
        image_path
    ).convert("RGB")

    mask = None

    if os.path.exists(
        mask_path
    ):

        mask = Image.open(
            mask_path
        ).convert("L")

    crop_image = make_lesion_crop(
        image,
        sample["bbox"]
    )

    try:

        (
            prediction,
            benign_probability,
            malignant_probability,
            full_tensor,
            crop_tensor
        ) = predict(
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

        gradcam_image = create_gradcam_overlay(
            image,
            cam
        )

    except Exception as error:

        st.error(
            "Sample analysis failed."
        )

        st.exception(error)

        st.stop()

    st.markdown(
        """
        <div class="section-header">

            <div class="section-title">
                Selected Case
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    m1, m2, m3 = st.columns(
        3,
        gap="medium"
    )

    with m1:

        st.markdown(
            f"""
            <div class="case-meta">

                <div class="case-meta-label">
                    Case ID
                </div>

                <div class="case-meta-value">
                    {sample["id"]}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with m2:

        st.markdown(
            f"""
            <div class="case-meta">

                <div class="case-meta-label">
                    Reference Class
                </div>

                <div class="case-meta-value">
                    {sample["label"]}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with m3:

        st.markdown(
            f"""
            <div class="case-meta">

                <div class="case-meta-label">
                    Histology
                </div>

                <div class="case-meta-value">
                    {sample["histology"]}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # AI RESULT
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="section-header">

            <div class="section-title">
                AI Assessment
            </div>

            <div class="section-subtitle">
                Model prediction and class probabilities.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    render_prediction(
        prediction,
        benign_probability,
        malignant_probability
    )

    # --------------------------------------------------------
    # EXPLANATION
    # --------------------------------------------------------

    render_explanation_section(
        image,
        gradcam_image,
        crop_image,
        mask=mask
    )


# ============================================================
# UPLOAD
# ============================================================

else:

    st.markdown(
        """
        <div class="workspace">

            <div class="workspace-title">
                Upload a Breast Ultrasound
            </div>

            <div class="workspace-text">
                Upload a PNG, JPG or JPEG ultrasound image.
                The system will generate an AI-assisted prediction,
                probabilities and a Grad-CAM visualization.
            </div>

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
        ],
        help=(
            "PNG, JPG and JPEG images are supported. "
            "Do not upload patient-identifying information."
        )
    )

    if uploaded_file is None:

        st.markdown(
            """
            <div class="upload-note">
                Research demonstration only ·
                Please remove or avoid patient-identifying information.
            </div>
            """,
            unsafe_allow_html=True
        )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        width, height = image.size

        left, right = st.columns(
            [1.25, 0.75],
            gap="large"
        )

        with left:

            st.image(
                image,
                caption="Uploaded Ultrasound",
                use_container_width=True
            )

        with right:

            st.markdown(
                """
                <div class="prediction-panel">

                    <div class="prediction-label">
                        Image Information
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.write(
                f"**Dimensions:** {width} × {height} px"
            )

            st.write(
                "**Format:** "
                f"{uploaded_file.type}"
            )

            st.write(
                "**Analysis:** "
                "Dual-view classification"
            )

            st.write(
                "**Explainability:** Grad-CAM"
            )

        st.markdown("")

        analyze = st.button(
            "Analyze Ultrasound",
            type="primary",
            use_container_width=True
        )

        if analyze:

            try:

                with st.spinner(
                    "Analyzing ultrasound..."
                ):

                    crop_image = (
                        generate_automatic_crop(
                            model,
                            image
                        )
                    )

                    (
                        prediction,
                        benign_probability,
                        malignant_probability,
                        full_tensor,
                        crop_tensor
                    ) = predict(
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

                    gradcam_image = (
                        create_gradcam_overlay(
                            image,
                            cam
                        )
                    )

                st.markdown(
                    """
                    <div class="section-header">

                        <div class="section-title">
                            AI Assessment
                        </div>

                        <div class="section-subtitle">
                            Model prediction and class probabilities.
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                render_prediction(
                    prediction,
                    benign_probability,
                    malignant_probability
                )

                render_explanation_section(
                    image,
                    gradcam_image,
                    crop_image,
                    mask=None
                )

            except Exception as error:

                st.error(
                    "Prediction failed."
                )

                st.exception(error)


# ============================================================
# TECHNICAL INFORMATION
# ============================================================

st.markdown("")

with st.expander(
    "Technical Model Information"
):

    t1, t2 = st.columns(
        2,
        gap="large"
    )

    with t1:

        st.markdown(
            "**Architecture**  \n"
            "Shared Dual EfficientNet-B3"
        )

        st.markdown(
            "**Classification**  \n"
            "Benign vs Malignant"
        )

        st.markdown(
            f"**Input resolution**  \n"
            "{IMAGE_SIZE} × {IMAGE_SIZE}"
        )

        st.markdown(
            f"**Decision threshold**  \n"
            "{THRESHOLD}"
        )

    with t2:

        st.markdown(
            f"**Inference device**  \n"
            "{device}"
        )

        st.markdown(
            "**Explainability**  \n"
            "Grad-CAM"
        )

        st.markdown(
            "**Sample localization**  \n"
            "BUS-BRA lesion BBOX"
        )

        st.markdown(
            "**Upload localization**  \n"
            "AI-generated attention crop"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        <strong>
            AI-Assisted Breast Ultrasound Analysis
        </strong>
        <br>

        Breast Ultrasound Research Prototype
        <br><br>

        Developed by <strong>Zara Ashraf</strong>
        <br>

        BS Medical Imaging Technology

    </div>
    """,
    unsafe_allow_html=True
)
