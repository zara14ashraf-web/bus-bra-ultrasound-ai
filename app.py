```python
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
# CUSTOM UI
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */

    .block-container {
        max-width: 1180px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .main {
        background-color: #ffffff;
    }

    /* ---------- HERO ---------- */

    .hero {
        text-align: center;
        padding: 1.8rem 1rem 1.2rem 1rem;
        margin-bottom: 1.2rem;
    }

    .hero-title {
        font-size: 2.7rem;
        font-weight: 700;
        color: #123b4a;
        letter-spacing: -1px;
        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        font-size: 1.15rem;
        color: #55717c;
        margin-bottom: 0.8rem;
    }

    .hero-description {
        max-width: 760px;
        margin: auto;
        color: #64748b;
        font-size: 0.96rem;
        line-height: 1.6;
    }

    /* ---------- DISCLAIMER ---------- */

    .disclaimer {
        background: #f8fafc;
        border: 1px solid #dbe4ea;
        border-radius: 12px;
        padding: 14px 18px;
        margin: 1rem 0 1.5rem 0;
        color: #475569;
        font-size: 0.88rem;
        line-height: 1.55;
    }

    .disclaimer strong {
        color: #123b4a;
    }

    /* ---------- SECTION ---------- */

    .section-title {
        font-size: 1.55rem;
        font-weight: 650;
        color: #183f4d;
        margin-top: 1.6rem;
        margin-bottom: 0.2rem;
    }

    .section-subtitle {
        color: #71808a;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }

    /* ---------- INFO CARDS ---------- */

    .info-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 18px;
        min-height: 92px;
    }

    .info-label {
        font-size: 0.76rem;
        color: #71808a;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 5px;
    }

    .info-value {
        font-size: 1rem;
        font-weight: 650;
        color: #183f4d;
    }

    /* ---------- WHY SECTION ---------- */

    .why-box {
        background: #f8fafc;
        border-left: 4px solid #2f7181;
        border-radius: 10px;
        padding: 17px 20px;
        color: #475569;
        line-height: 1.65;
        font-size: 0.92rem;
        margin-top: 0.7rem;
    }

    /* ---------- SAMPLE CARDS ---------- */

    .sample-title {
        font-size: 0.92rem;
        font-weight: 650;
        color: #183f4d;
        margin-top: 0.35rem;
    }

    .sample-label {
        font-size: 0.78rem;
        color: #71808a;
    }

    /* ---------- RESULT ---------- */

    .result-card {
        background: #f8fafc;
        border: 1px solid #dbe4ea;
        border-radius: 14px;
        padding: 22px;
        text-align: center;
        margin: 1rem 0 1.2rem 0;
    }

    .result-label {
        color: #71808a;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    .result-value {
        font-size: 2rem;
        font-weight: 750;
        margin-top: 4px;
    }

    .probability-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 15px 18px;
        margin-bottom: 10px;
    }

    .probability-name {
        font-size: 0.86rem;
        color: #475569;
        margin-bottom: 5px;
    }

    .probability-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #183f4d;
    }

    /* ---------- UPLOAD ---------- */

    .upload-note {
        text-align: center;
        color: #71808a;
        font-size: 0.82rem;
        margin-top: -0.4rem;
    }

    /* ---------- SIDEBAR ---------- */

    .sidebar-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #183f4d;
        margin-bottom: 0.2rem;
    }

    .sidebar-heading {
        font-size: 0.9rem;
        font-weight: 650;
        color: #183f4d;
        margin-top: 1.1rem;
        margin-bottom: 0.3rem;
    }

    .sidebar-text {
        color: #64748b;
        font-size: 0.82rem;
        line-height: 1.55;
    }

    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.78rem;
        margin-top: 1.8rem;
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

    with st.spinner("Preparing SonoInsight AI model..."):

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

    return Image.fromarray(result)


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
# AUTOMATIC CAM CROP
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

    ys, xs = np.where(active)

    x1 = xs.min()
    x2 = xs.max()

    y1 = ys.min()
    y2 = ys.max()

    roi_width = x2 - x1 + 1
    roi_height = y2 - y1 + 1

    pad_x = int(
        roi_width * padding_ratio
    )

    pad_y = int(
        roi_height * padding_ratio
    )

    x1 -= pad_x
    y1 -= pad_y
    x2 += pad_x
    y2 += pad_y

    scale_x = (
        image_width / IMAGE_SIZE
    )

    scale_y = (
        image_height / IMAGE_SIZE
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

    if x2 <= x1 or y2 <= y1:

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

        <div class="hero-description">
            A research prototype exploring dual-view deep learning
            for benign and malignant breast ultrasound classification,
            supported by visual explainability through Grad-CAM.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DISCLAIMER — VISIBLE AT START
# ============================================================

st.markdown(
    """
    <div class="disclaimer">

        <strong>Research & Educational Prototype</strong><br>

        SonoInsight AI is developed for research and educational
        demonstration only. It is not a medical device and should
        not be used to diagnose, exclude, or guide treatment of
        breast disease. AI predictions should never replace
        assessment by a qualified healthcare professional.

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-title">Model Overview</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'A dual-view architecture designed to examine both global '
    'ultrasound appearance and a lesion-focused view.'
    '</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.markdown(
        """
        <div class="info-card">
            <div class="info-label">Architecture</div>
            <div class="info-value">
                Dual EfficientNet-B3
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:

    st.markdown(
        """
        <div class="info-card">
            <div class="info-label">Task</div>
            <div class="info-value">
                Benign vs Malignant
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:

    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-label">Input Resolution</div>
            <div class="info-value">
                {IMAGE_SIZE} × {IMAGE_SIZE}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c4:

    st.markdown(
        """
        <div class="info-card">
            <div class="info-label">Explainability</div>
            <div class="info-value">
                Grad-CAM
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# WHY WAS IT DEVELOPED?
# ============================================================

st.markdown(
    '<div class="section-title">Why was SonoInsight AI developed?</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="why-box">

    As a Medical Imaging Technologist, I developed SonoInsight AI
    to explore how artificial intelligence can complement medical
    imaging workflows and make breast ultrasound analysis more
    transparent.

    Breast ultrasound images can demonstrate substantial variation
    in lesion appearance, making consistent image-based assessment
    challenging. This project investigates whether a dual-view
    deep learning approach can learn from both the overall
    ultrasound image and a focused lesion representation.

    The addition of Grad-CAM provides a visual explanation of
    regions that contributed to the model's prediction, making
    the system more interpretable for research and educational
    purposes.

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">🩺 SonoInsight AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-text">
        Breast ultrasound research prototype using
        dual-view deep learning and Grad-CAM.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        '<div class="sidebar-heading">What does it analyze?</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-text">
        The model estimates whether an ultrasound lesion
        is more consistent with a benign or malignant class.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-heading">Model Classes</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-text">
        🟢 <strong>Benign</strong><br>
        🔴 <strong>Malignant</strong>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-heading">Dual-View Approach</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-text">
        The architecture combines information from the
        full ultrasound image with a lesion-focused view.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-heading">Explainability</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-text">
        Grad-CAM highlights image regions that contributed
        to the selected model prediction.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-heading">Important</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-text">
        This application is a research prototype.
        AI output must not be interpreted as a clinical diagnosis.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        '<div class="sidebar-heading">System</div>',
        unsafe_allow_html=True
    )

    st.caption(
        f"Device: {device}"
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
    '<div class="section-title">Analyze Breast Ultrasound</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Explore representative cases or upload an ultrasound image '
    'for an AI-assisted research analysis.'
    '</div>',
    unsafe_allow_html=True
)


analysis_mode = st.radio(
    "Analysis mode",
    [
        "Explore Sample Cases",
        "Upload Your Ultrasound"
    ],
    horizontal=True,
    label_visibility="collapsed"
)


# ============================================================
# SAMPLE CASES
# ============================================================

if analysis_mode == "Explore Sample Cases":

    st.markdown(
        "#### Representative BUS-BRA Cases"
    )

    st.caption(
        "Select a case to explore the reference lesion annotation, "
        "model prediction, and Grad-CAM explanation."
    )

    sample_names = list(SAMPLES.keys())

    sample_cols = st.columns(4)

    selected_sample = None

    for index, sample_name in enumerate(sample_names):

        sample = SAMPLES[sample_name]

        image_path = os.path.join(
            BASE_DIR,
            sample["image"]
        )

        with sample_cols[index]:

            if os.path.exists(image_path):

                preview = Image.open(
                    image_path
                ).convert("RGB")

                st.image(
                    preview,
                    use_container_width=True
                )

            st.markdown(
                f"""
                <div class="sample-title">
                    {sample_name}
                </div>
                <div class="sample-label">
                    {sample["id"]}
                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button(
                "Explore",
                key=f"sample_{index}",
                use_container_width=True
            ):

                selected_sample = sample_name

    if "selected_sample" not in st.session_state:

        st.session_state.selected_sample = sample_names[0]

    if selected_sample is not None:

        st.session_state.selected_sample = selected_sample

    selected_sample = st.session_state.selected_sample

    sample = SAMPLES[selected_sample]

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
            f"Sample image not found: {sample['image']}"
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

    st.divider()

    # --------------------------------------------------------
    # CASE INFORMATION
    # --------------------------------------------------------

    st.markdown(
        "#### Case Information"
    )

    i1, i2, i3 = st.columns(3)

    with i1:

        st.metric(
            "Case ID",
            sample["id"]
        )

    with i2:

        st.metric(
            "Reference Class",
            sample["label"]
        )

    with i3:

        st.metric(
            "Histology",
            sample["histology"]
        )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    st.markdown(
        "#### AI Assessment"
    )

    if prediction == "Malignant":

        result_color = "#b42318"

    else:

        result_color = "#177245"

    st.markdown(
        f"""
        <div class="result-card">

            <div class="result-label">
                Model Prediction
            </div>

            <div class="result-value"
                 style="color:{result_color};">
                {prediction}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    p1, p2 = st.columns(2)

    with p1:

        st.markdown(
            f"""
            <div class="probability-card">
                <div class="probability-name">
                    🟢 Benign Probability
                </div>
                <div class="probability-value">
                    {benign_probability * 100:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with p2:

        st.markdown(
            f"""
            <div class="probability-card">
                <div class="probability-name">
                    🔴 Malignant Probability
                </div>
                <div class="probability-value">
                    {malignant_probability * 100:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # VISUAL EXPLANATION
    # --------------------------------------------------------

    st.markdown(
        "#### Visual Explanation"
    )

    st.caption(
        "Compare the reference lesion annotation with the "
        "regions highlighted by the model."
    )

    v1, v2, v3 = st.columns(3)

    with v1:

        if mask is not None:

            mask_overlay = create_mask_overlay(
                image,
                mask
            )

            st.image(
                mask_overlay,
                caption="Ground-Truth Lesion Mask",
                use_container_width=True
            )

        else:

            st.image(
                image,
                caption="Original Ultrasound",
                use_container_width=True
            )

    with v2:

        st.image(
            gradcam_image,
            caption="Grad-CAM Attention",
            use_container_width=True
        )

    with v3:

        st.image(
            crop_image,
            caption="Lesion-Focused Model View",
            use_container_width=True
        )


# ============================================================
# UPLOAD ANALYSIS
# ============================================================

else:

    st.markdown(
        "#### Upload Your Ultrasound"
    )

    st.caption(
        "Upload a breast ultrasound image for a research "
        "demonstration of the model's prediction and visual attention."
    )

    uploaded_file = st.file_uploader(
        "Choose an ultrasound image",
        type=[
            "png",
            "jpg",
            "jpeg"
        ],
        help="PNG, JPG and JPEG images are supported.",
    )

    if uploaded_file is None:

        st.markdown(
            """
            <div class="upload-note">
                For research demonstration only.
                Please do not upload patient-identifying information.
            </div>
            """,
            unsafe_allow_html=True
        )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        width, height = image.size

        st.image(
            image,
            caption="Uploaded Ultrasound",
            width=520
        )

        st.caption(
            f"Image dimensions: {width} × {height} pixels"
        )

        if st.button(
            "🔍 Analyze Ultrasound",
            type="primary",
            use_container_width=True
        ):

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

                st.divider()

                # ------------------------------------------------
                # RESULT
                # ------------------------------------------------

                st.markdown(
                    "#### AI Assessment"
                )

                if prediction == "Malignant":

                    result_color = "#b42318"

                else:

                    result_color = "#177245"

                st.markdown(
                    f"""
                    <div class="result-card">

                        <div class="result-label">
                            Model Prediction
                        </div>

                        <div class="result-value"
                             style="color:{result_color};">
                            {prediction}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                p1, p2 = st.columns(2)

                with p1:

                    st.markdown(
                        f"""
                        <div class="probability-card">
                            <div class="probability-name">
                                🟢 Benign Probability
                            </div>
                            <div class="probability-value">
                                {benign_probability * 100:.1f}%
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with p2:

                    st.markdown(
                        f"""
                        <div class="probability-card">
                            <div class="probability-name">
                                🔴 Malignant Probability
                            </div>
                            <div class="probability-value">
                                {malignant_probability * 100:.1f}%
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # ------------------------------------------------
                # VISUAL EXPLANATION
                # ------------------------------------------------

                st.markdown(
                    "#### Visual Explanation"
                )

                st.caption(
                    "The lesion-focused view is generated automatically "
                    "using model attention."
                )

                v1, v2, v3 = st.columns(3)

                with v1:

                    st.image(
                        image,
                        caption="Uploaded Ultrasound",
                        use_container_width=True
                    )

                with v2:

                    st.image(
                        gradcam_image,
                        caption="Grad-CAM Attention",
                        use_container_width=True
                    )

                with v3:

                    st.image(
                        crop_image,
                        caption="AI-Generated Lesion View",
                        use_container_width=True
                    )

                st.info(
                    "Reference lesion masks are available only for "
                    "the representative BUS-BRA sample cases. "
                    "Uploaded images do not include ground-truth annotations."
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
    "Technical Model Information"
):

    m1, m2 = st.columns(2)

    with m1:

        st.write(
            "**Architecture:** Shared Dual EfficientNet-B3"
        )

        st.write(
            "**Classification:** Benign vs Malignant"
        )

        st.write(
            f"**Input resolution:** {IMAGE_SIZE} × {IMAGE_SIZE}"
        )

        st.write(
            f"**Decision threshold:** {THRESHOLD}"
        )

    with m2:

        st.write(
            f"**Inference device:** {device}"
        )

        st.write(
            "**Explainability:** Grad-CAM"
        )

        st.write(
            "**Sample localization:** BUS-BRA BBOX"
        )

        st.write(
            "**Upload localization:** AI-generated attention crop"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        SonoInsight AI · Breast Ultrasound Research Prototype
    </div>
    """,
    unsafe_allow_html=True
)
```
