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
# SIMPLE CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1200px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .hero-title {
        text-align: center;
        font-size: 2.55rem;
        font-weight: 750;
        color: #173f4d;
        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        text-align: center;
        font-size: 1.05rem;
        color: #607984;
        margin-bottom: 0.75rem;
    }

    .hero-description {
        max-width: 780px;
        margin: 0 auto;
        text-align: center;
        color: #71838b;
        font-size: 0.92rem;
        line-height: 1.65;
    }

    .small-muted {
        color: #71838b;
        font-size: 0.86rem;
    }

    .result-number {
        font-size: 2.1rem;
        font-weight: 750;
        margin-top: 0.2rem;
    }

    .footer {
        text-align: center;
        color: #8a989e;
        font-size: 0.75rem;
        line-height: 1.6;
        padding-top: 1.5rem;
        margin-top: 2rem;
        border-top: 1px solid #e5eaed;
    }

    [data-testid="stSidebar"] {
        background-color: #f7fafb;
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
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "SharedDualEfficientNetB3"

IMAGE_SIZE = 300

THRESHOLD = 0.52

CROP_MARGIN = 0.25

MEAN = [
    0.485,
    0.456,
    0.406,
]

STD = [
    0.229,
    0.224,
    0.225,
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
        "Preparing the AI model for the first use..."
    ):

        response = requests.get(
            HF_CHECKPOINT_URL,
            stream=True,
            timeout=300,
        )

        response.raise_for_status()

        with open(
            CHECKPOINT_PATH,
            "wb",
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
        weights_only=False,
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
        strict=True,
    )

    model = model.to(device)

    model.eval()

    return model


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=MEAN,
            std=STD,
        ),
    ]
)


# ============================================================
# LESION CROP
# ============================================================

def make_lesion_crop(
    image,
    bbox,
    margin=CROP_MARGIN,
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
        x - pad_x,
    )

    y1 = max(
        0,
        y - pad_y,
    )

    x2 = min(
        image_width,
        x + width + pad_x,
    )

    y2 = min(
        image_height,
        y + height + pad_y,
    )

    return image.crop(
        (x1, y1, x2, y2)
    )


# ============================================================
# MASK OVERLAY
# ============================================================

def create_mask_overlay(
    image,
    mask,
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
        255,
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
        target_layer,
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
        output,
    ):

        self.activations = output

    def _save_gradient(
        self,
        module,
        grad_input,
        grad_output,
    ):

        self.gradients = grad_output[0]

    def generate(
        self,
        full_tensor,
        crop_tensor,
        target_class,
    ):

        self.model.zero_grad(
            set_to_none=True
        )

        logits = self.model(
            full_tensor,
            crop_tensor,
        )

        score = logits[
            0,
            target_class,
        ]

        score.backward()

        activations = self.activations
        gradients = self.gradients

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        cam = (
            weights * activations
        ).sum(
            dim=1,
            keepdim=True,
        )

        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=(
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[
            0,
            0,
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
    predicted_class,
):

    target_layer = (
        model.full_branch.blocks[-1]
    )

    gradcam = GradCAM(
        model,
        target_layer,
    )

    try:

        cam = gradcam.generate(
            full_tensor,
            crop_tensor,
            predicted_class,
        )

    finally:

        gradcam.remove_hooks()

    return cam


# ============================================================
# GRAD-CAM OVERLAY
# ============================================================

def create_gradcam_overlay(
    image,
    cam,
):

    image_resized = image.resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
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
            3,
        ),
        dtype=np.float32,
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
        1,
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
    crop_image,
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
            crop_tensor,
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
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
# AI ASSESSMENT
# ============================================================

def display_ai_assessment(
    prediction,
    benign_probability,
    malignant_probability,
):

    # --------------------------------------------------------
    # PROBABILITY SEPARATION
    # --------------------------------------------------------

    probability_separation = abs(
        benign_probability - malignant_probability
    )

    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    if probability_separation < 0.10:

        assessment_level = "Low model separation"

        assessment_message = (
            "The model assigns similar probabilities to both "
            "classes for this image. This represents an uncertain "
            "model output and should be interpreted cautiously."
        )

    elif probability_separation < 0.20:

        assessment_level = "Moderate model separation"

        assessment_message = (
            "The model shows a preference for one class, but "
            "the alternative class remains relatively close."
        )

    else:

        assessment_level = "Clearer model separation"

        assessment_message = (
            "The model shows a more distinct probability difference "
            "between the two classes."
        )

    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    if prediction == "Benign":

        st.success(
            "### Model Prediction: BENIGN"
        )

    else:

        st.error(
            "### Model Prediction: MALIGNANT"
        )

    # --------------------------------------------------------
    # MODEL OUTPUT INTERPRETATION
    # --------------------------------------------------------

    if probability_separation < 0.10:

        st.warning(
            f"**{assessment_level}**"
        )

    elif probability_separation < 0.20:

        st.warning(
            f"**{assessment_level}**"
        )

    else:

        st.success(
            f"**{assessment_level}**"
        )

    st.caption(
        assessment_message
    )

    st.write("")

    # --------------------------------------------------------
    # MODEL OUTPUT DISTRIBUTION
    # --------------------------------------------------------

    with st.expander(
    "View model probability distribution",
    expanded=False,
):

    st.caption(
        "The following values show the model's raw output "
        "distribution for this image."
    )

    prob1, prob2 = st.columns(
        2,
        gap="large",
    )

    with prob1:

        st.metric(
            "Benign",
            f"{benign_probability * 100:.1f}%",
        )

        st.progress(
            benign_probability
        )

    with prob2:

        st.metric(
            "Malignant",
            f"{malignant_probability * 100:.1f}%",
        )

        st.progress(
            malignant_probability
        )

    # --------------------------------------------------------
    # INTERPRETATION NOTE
    # --------------------------------------------------------

    st.caption(
        "These percentages represent the model's output "
        "distribution for this image. They do not represent "
        "diagnostic accuracy, disease probability, or clinical "
        "certainty."
    )

    st.caption(
        f"Probability separation: "
        f"{probability_separation * 100:.1f} percentage points"
    )

# ============================================================
# BBOX FROM CAM
# ============================================================

def bbox_from_cam(
    cam,
    original_size,
    threshold_ratio=0.55,
    padding_ratio=0.20,
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
            image_height,
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
            x1 * scale_x,
        )
    )

    y1 = int(
        max(
            0,
            y1 * scale_y,
        )
    )

    x2 = int(
        min(
            image_width,
            x2 * scale_x,
        )
    )

    y2 = int(
        min(
            image_height,
            y2 * scale_y,
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
            image_height,
        ]

    return [
        x1,
        y1,
        x2 - x1,
        y2 - y1,
    ]


# ============================================================
# AUTOMATIC UPLOAD CROP
# ============================================================

def generate_automatic_crop(
    model,
    image,
):

    full_tensor = transform(
        image
    ).unsqueeze(
        0
    ).to(device)

    with torch.no_grad():

        logits = model(
            full_tensor,
            full_tensor,
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
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
        initial_class,
    )

    bbox = bbox_from_cam(
        cam,
        image.size,
    )

    crop = make_lesion_crop(
        image,
        bbox,
        margin=0.0,
    )

    return crop


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🩺 Breast Ultrasound AI")

    st.caption(
        "AI-assisted medical imaging research project"
    )

    st.divider()

    st.subheader("About the Project")

    st.write(
        "This application explores deep learning for "
        "breast ultrasound lesion classification."
    )

    st.write(
        "The model uses a dual-view approach, combining "
        "the complete ultrasound image with a "
        "lesion-focused representation."
    )

    st.subheader("Model Classes")

    st.write("🟢 **Benign**")
    st.write("🔴 **Malignant**")

    st.subheader("How It Works")

    st.write(
        "The uploaded ultrasound is processed by the "
        "trained dual-view model. The system produces "
        "a class prediction and probability distribution."
    )

    st.write(
        "Grad-CAM is then used to visualize image regions "
        "associated with the model's prediction."
    )

    st.subheader("Explainability")

    st.write(
        "Grad-CAM provides a visual indication of regions "
        "that contributed to the selected prediction."
    )

    st.divider()

    st.subheader("Developed By")

    st.markdown(
        "**Zara Ashraf**"
    )

    st.caption(
        "BS Medical Imaging Technology"
    )

    st.divider()

    st.caption(
        "Research & educational prototype."
    )

    st.caption(
        "Not intended for clinical diagnosis."
    )


# ============================================================
# HERO
# ============================================================
st.markdown(
    """
    <h1 style="
        text-align: center;
        color: #173f4d;
        font-size: 2.45rem;
        font-weight: 750;
        margin: 0 0 0.35rem 0;
        line-height: 1.2;
    ">
        AI-Assisted Breast Ultrasound Analysis
    </h1>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p style="
        text-align: center;
        color: #607984;
        font-size: 1.05rem;
        margin: 0 0 0.55rem 0;
    ">
        Deep Learning for Breast Lesion Classification
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p style="
        text-align: center;
        color: #71838b;
        font-size: 0.92rem;
        line-height: 1.6;
        max-width: 760px;
        margin: 0 auto 1.5rem auto;
    ">
        A research prototype exploring dual-view deep learning
        for benign and malignant breast ultrasound classification,
        with probability-based predictions and Grad-CAM
        visual explainability.
    </p>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DISCLAIMER
# ============================================================

st.info(
    """
    **Research & Educational Prototype**

    This application is developed for research and educational
    demonstration only. It is not a medical device and should
    not be used to diagnose, exclude, or guide treatment of
    breast disease. AI predictions should not replace assessment
    by a qualified healthcare professional.
    """
)


# ============================================================
# MODEL OVERVIEW
# ============================================================

with st.expander(
    "🧠 Model Overview",
    expanded=False,
):

    st.write(
        "The model uses a dual-view deep learning architecture "
        "designed to examine both the overall ultrasound image "
        "and a lesion-focused representation."
    )

    st.write("")

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Architecture",
            "Dual EfficientNet-B3",
        )

    with c2:

        st.metric(
            "Task",
            "Benign vs Malignant",
        )

    with c3:

        st.metric(
            "Input",
            "300 × 300",
        )

    with c4:

        st.metric(
            "Explainability",
            "Grad-CAM",
        )


# ============================================================
# WHY WAS THIS DEVELOPED?
# ============================================================

with st.expander(
    "🔬 Why was this developed?",
    expanded=False,
):

    st.write(
        "As a Medical Imaging Technologist, I developed this "
        "project to explore how artificial intelligence can "
        "complement medical imaging workflows and support "
        "research in breast ultrasound analysis."
    )

    st.write(
        "Breast ultrasound images can demonstrate considerable "
        "variation in lesion appearance. This project investigates "
        "whether a dual-view deep learning approach can learn from "
        "both the overall ultrasound image and a focused lesion view."
    )

    st.write(
        "The aim is not to replace radiologists or medical imaging "
        "professionals. Instead, the system explores AI as an "
        "assistive research tool that can provide an additional "
        "image-based perspective together with visual explanation "
        "through Grad-CAM."
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
# ANALYSIS AREA
# ============================================================

st.divider()

st.markdown(
    "### Analyze Breast Ultrasound"
)

st.caption(
    "Upload an ultrasound image or explore a representative "
    "sample case to see how the model analyzes breast lesions."
)

mode = st.radio(
    "Choose an analysis method",
    [
        "Upload Ultrasound",
        "Explore Sample Cases",
    ],
    horizontal=True,
    label_visibility="collapsed",
)


# ============================================================
# UPLOAD MODE
# ============================================================

if mode == "Upload Ultrasound":

    st.write("")

    st.markdown(
        "#### Upload an ultrasound image"
    )

    st.caption(
        "Choose a breast ultrasound image to begin the "
        "AI-assisted research analysis."
    )

    uploaded_file = st.file_uploader(
        "Upload ultrasound",
        type=[
            "png",
            "jpg",
            "jpeg",
        ],
        label_visibility="collapsed",
        help=(
            "Supported formats: PNG, JPG and JPEG. "
            "Please remove patient-identifying information "
            "before uploading."
        ),
    )

    # --------------------------------------------------------
    # NO IMAGE
    # --------------------------------------------------------

    if uploaded_file is None:

        st.info(
            "PNG, JPG and JPEG images are supported. "
            "Please remove patient-identifying information "
            "before uploading."
        )

        st.caption(
            "Research & educational prototype — "
            "not intended for clinical diagnosis."
        )

    # --------------------------------------------------------
    # IMAGE UPLOADED
    # --------------------------------------------------------

    else:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        width, height = image.size

        st.write("")

        # ----------------------------------------------------
        # IMAGE PREVIEW
        # ----------------------------------------------------

        preview_col, info_col = st.columns(
            [2.3, 1],
            gap="large",
        )

        with preview_col:

            st.image(
                image,
                caption="Uploaded Ultrasound",
                use_container_width=True,
            )

        with info_col:

            st.markdown(
                "#### Image Information"
            )

            st.markdown(
                f"""
                <div style="
                    font-size: 1.35rem;
                    font-weight: 700;
                    color: #173f4d;
                    margin: 0.25rem 0 0.15rem 0;
                ">
                    {width} × {height} px
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.caption(
                "Image dimensions"
            )

            st.write("")

            st.markdown(
                "**Format**"
            )

            st.caption(
                "Ultrasound image"
            )

            st.write("")

            st.caption(
                "Please ensure patient-identifying information "
                "has been removed before analysis."
            )

        st.write("")

        # ----------------------------------------------------
        # ANALYZE BUTTON
        # ----------------------------------------------------

        analyze = st.button(
            "🔍 Analyze Ultrasound",
            type="primary",
            use_container_width=True,
        )

        if analyze:

            try:

                with st.spinner(
                    "Analyzing ultrasound..."
                ):

                    # ----------------------------------------
                    # AUTOMATIC LESION CROP
                    # ----------------------------------------

                    crop_image = (
                        generate_automatic_crop(
                            model,
                            image,
                        )
                    )

                    # ----------------------------------------
                    # PREDICTION
                    # ----------------------------------------

                    (
                        prediction,
                        benign_probability,
                        malignant_probability,
                        full_tensor,
                        crop_tensor,
                    ) = predict(
                        model,
                        image,
                        crop_image,
                    )

                    predicted_class = (
                        1
                        if prediction == "Malignant"
                        else 0
                    )

                    # ----------------------------------------
                    # GRAD-CAM
                    # ----------------------------------------

                    cam = generate_gradcam(
                        model,
                        full_tensor,
                        crop_tensor,
                        predicted_class,
                    )

                    gradcam_image = (
                        create_gradcam_overlay(
                            image,
                            cam,
                        )
                    )

                # =================================================
                # AI ASSESSMENT
                # =================================================

                st.divider()

                st.subheader(
                    "AI Assessment"
                )

                display_ai_assessment(
                    prediction,
                    benign_probability,
                    malignant_probability,
                )

                # =================================================
                # VISUAL EXPLANATION
                # =================================================

                st.write("")

                st.subheader(
                    "Visual Explanation"
                )

                st.caption(
                    "Compare the original ultrasound, the "
                    "automatically generated lesion-focused view, "
                    "and the regions highlighted by Grad-CAM."
                )

                visual1, visual2, visual3 = st.columns(
                    3,
                    gap="medium",
                )

                with visual1:

                    st.image(
                        image,
                        caption="Original Ultrasound",
                        use_container_width=True,
                    )

                with visual2:

                    st.image(
                        crop_image,
                        caption="AI-Generated Lesion View",
                        use_container_width=True,
                    )

                with visual3:

                    st.image(
                        gradcam_image,
                        caption="Grad-CAM Attention",
                        use_container_width=True,
                    )

                st.info(
                    "For uploaded images, the focused view is generated "
                    "automatically based on model attention. It is intended "
                    "as an additional representation for analysis and does "
                    "not represent a definitive lesion segmentation."
                )

                # ---------------------------------------------
                # DISCLAIMER
                # ---------------------------------------------

                st.caption(
                    "Research & educational prototype. "
                    "AI predictions are not a clinical diagnosis "
                    "and should not replace assessment by a "
                    "qualified healthcare professional."
                )

            except Exception as error:

                st.error(
                    "Prediction failed."
                )

                st.exception(error)


# ============================================================
# SAMPLE CASE MODE
# ============================================================


# ============================================================
# SAMPLE CASE MODE
# ============================================================

else:

    st.write("")

    st.caption(
        "Explore representative ultrasound cases "
        "and see how the model analyzes each lesion."
    )

    # --------------------------------------------------------
    # SAMPLE CASE SELECTOR
    # --------------------------------------------------------

    sample_names = list(SAMPLES.keys())

    if "selected_sample" not in st.session_state:
        st.session_state.selected_sample = sample_names[0]

    st.markdown(
        "#### Representative Sample Cases"
    )

    st.caption(
        "Select a sample case to explore its AI-assisted analysis."
    )

    sample_cols = st.columns(
        4,
        gap="medium",
    )

    for i, sample_name in enumerate(sample_names):

        with sample_cols[i]:

            if st.button(
                f"Sample {i + 1}",
                key=f"sample_button_{i}",
                use_container_width=True,
            ):

                st.session_state.selected_sample = sample_name

    selected_sample_name = (
        st.session_state.selected_sample
    )

    sample = SAMPLES[
        selected_sample_name
    ]
    # --------------------------------------------------------
    # SELECTED SAMPLE
    # --------------------------------------------------------

    selected_sample_name = (
        st.session_state.selected_sample
    )

    sample = SAMPLES[
        selected_sample_name
    ]

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    image_path = os.path.join(
        BASE_DIR,
        sample["image"],
    )

    mask_path = os.path.join(
        BASE_DIR,
        sample["mask"],
    )

    if not os.path.exists(image_path):

        st.error(
            "Selected sample image could not be found."
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

    # --------------------------------------------------------
    # LESION-FOCUSED VIEW
    # --------------------------------------------------------

    crop_image = make_lesion_crop(
        image,
        sample["bbox"],
    )

    # --------------------------------------------------------
    # MODEL ANALYSIS
    # --------------------------------------------------------

    try:

        (
            prediction,
            benign_probability,
            malignant_probability,
            full_tensor,
            crop_tensor,
        ) = predict(
            model,
            image,
            crop_image,
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
            predicted_class,
        )

        gradcam_image = create_gradcam_overlay(
            image,
            cam,
        )

    except Exception as error:

        st.error(
            "Sample analysis failed."
        )

        st.exception(error)

        st.stop()

    # --------------------------------------------------------
    # SELECTED CASE LABEL
    # --------------------------------------------------------

    st.write("")

    st.markdown(
        f"### {selected_sample_name}"
    )

    # --------------------------------------------------------
    # VISUAL EXPLANATION
    # --------------------------------------------------------

    st.write("")

    st.subheader(
        "Visual Explanation"
    )

    st.caption(
        "Compare the original ultrasound, the lesion-focused "
        "view, and the regions highlighted by Grad-CAM."
    )

    visual1, visual2, visual3 = st.columns(
        3,
        gap="medium",
    )

    with visual1:

        st.image(
            image,
            caption="Original Ultrasound",
            use_container_width=True,
        )

    with visual2:

        st.image(
            crop_image,
            caption="Lesion-Focused View",
            use_container_width=True,
        )

    with visual3:

        st.image(
            gradcam_image,
            caption="Grad-CAM Attention",
            use_container_width=True,
        )

    # --------------------------------------------------------
    # AI ASSESSMENT + REFERENCE
    # --------------------------------------------------------

    st.write("")

    assessment_col, reference_col = st.columns(
        [1.35, 1],
        gap="large",
    )

    with assessment_col:

        st.subheader(
            "AI Assessment"
        )

        if prediction == "Benign":

            st.success(
                "Model prediction: **BENIGN**"
            )

        else:

            st.error(
                "Model prediction: **MALIGNANT**"
            )

        st.write("")

        prob1, prob2 = st.columns(2)

        with prob1:

            st.metric(
                "Benign",
                f"{benign_probability * 100:.1f}%",
            )

            st.progress(
                benign_probability
            )

        with prob2:

            st.metric(
                "Malignant",
                f"{malignant_probability * 100:.1f}%",
            )

            st.progress(
                malignant_probability
            )

        st.caption(
    "The displayed percentages represent the model's output "
    "distribution for this image. They do not represent "
    "diagnostic accuracy or clinical certainty."
)

    with reference_col:

        st.subheader(
            "Reference Information"
        )

        st.caption(
            "DATASET REFERENCE"
        )

        st.markdown(
            f"**{sample['label']}**"
        )

        st.caption(
            "HISTOLOGY"
        )

        st.markdown(
            f"**{sample['histology']}**"
        )

        st.caption(
            "Reference information is shown only "
            "for dataset comparison."
        )

    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    st.write("")

    st.info(
        """
        **Research & Educational Prototype**

        This sample case is provided for research and
        educational demonstration only. The reference
        information belongs to the dataset and the AI
        prediction is not a clinical diagnosis. This
        application should not be used to diagnose,
        exclude, or guide treatment of breast disease.
        AI predictions should not replace assessment by
        a qualified healthcare professional.
        """
    )

    st.caption(
        "The lesion-focused view uses the reference lesion "
        "region available for this representative BUS-BRA case. "
        "Grad-CAM indicates model attention and is not a "
        "definitive lesion segmentation."
    )

# ============================================================
# TECHNICAL INFORMATION
# ============================================================

st.divider()

with st.expander(
    "Technical Model Information"
):

    technical1, technical2 = st.columns(
        2
    )

    with technical1:

        st.write(
            "**Architecture:** "
            "Shared Dual EfficientNet-B3"
        )

        st.write(
            "**Task:** "
            "Binary breast lesion classification"
        )

        st.write(
            "**Classes:** "
            "Benign and Malignant"
        )

        st.write(
            f"**Input resolution:** "
            f"{IMAGE_SIZE} × {IMAGE_SIZE}"
        )

    with technical2:

        st.write(
            f"**Decision threshold:** "
            f"{THRESHOLD}"
        )

        st.write(
            f"**Inference device:** "
            f"{device}"
        )

        st.write(
            "**Explainability:** Grad-CAM"
        )

        st.write(
            "**Model input:** Full image + lesion-focused view"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <b>AI-Assisted Breast Ultrasound Analysis</b><br>
        Breast Ultrasound Research Prototype<br><br>
        Developed by <b>Zara Ashraf</b><br>
        BS Medical Imaging Technology
    </div>
    """,
    unsafe_allow_html=True,
)
