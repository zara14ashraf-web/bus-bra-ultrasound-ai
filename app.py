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
)


# ============================================================
# BASIC STYLING
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1 {
        padding-bottom: 0.2rem;
    }

    .small-text {
        color: #64748b;
        font-size: 0.9rem;
    }

    .disclaimer {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px 16px;
        color: #475569;
        font-size: 0.85rem;
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

    with st.spinner("Downloading trained AI model..."):

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
# BBOX-BASED LESION CROP
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
# GROUND-TRUTH MASK OVERLAY
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
# AUTOMATIC ATTENTION-BASED CROP
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

    roi_width = x2 - x1 + 1
    roi_height = y2 - y1 + 1

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

    # Initial localization pass.
    # The same image is used temporarily for
    # both model branches.

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
# HEADER
# ============================================================

st.title(
    "🩺 SonoInsight AI"
)

st.caption(
    "AI-Assisted Breast Ultrasound Analysis"
)

st.write(
    "Dual-view deep learning for benign vs malignant "
    "breast ultrasound classification with visual explainability."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Analysis")

    analysis_mode = st.radio(
        "Choose mode",
        [
            "Explore Sample Cases",
            "Analyze Your Ultrasound"
        ]
    )

    st.divider()

    st.subheader("About")

    st.caption(
        "A research prototype demonstrating "
        "dual-view deep learning and Grad-CAM "
        "explainability for breast ultrasound."
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
# SAMPLE CASES
# ============================================================

if analysis_mode == "Explore Sample Cases":

    st.header(
        "Explore Sample Cases"
    )

    st.caption(
        "Representative BUS-BRA cases with reference "
        "lesion annotations and model explainability."
    )

    selected_sample = st.selectbox(
        "Select a case",
        list(SAMPLES.keys())
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

    bbox = sample["bbox"]

    # Dataset BBOX -> model crop

    crop_image = make_lesion_crop(
        image,
        bbox
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

    # --------------------------------------------------------
    # CASE INFORMATION
    # --------------------------------------------------------

    st.subheader(
        "Case Information"
    )

    info1, info2, info3 = st.columns(3)

    with info1:

        st.metric(
            "Case ID",
            sample["id"]
        )

    with info2:

        st.metric(
            "Reference Pathology",
            sample["label"]
        )

    with info3:

        st.metric(
            "Histology",
            sample["histology"]
        )

    # --------------------------------------------------------
    # AI PREDICTION
    # --------------------------------------------------------

    st.subheader(
        "AI Prediction"
    )

    if prediction == "Malignant":

        st.error(
            f"Prediction: {prediction}"
        )

    else:

        st.success(
            f"Prediction: {prediction}"
        )

    p1, p2 = st.columns(2)

    with p1:

        st.metric(
            "Benign probability",
            f"{benign_probability * 100:.1f}%"
        )

    with p2:

        st.metric(
            "Malignant probability",
            f"{malignant_probability * 100:.1f}%"
        )

    # --------------------------------------------------------
    # THREE VISUALS
    # --------------------------------------------------------

    st.subheader(
        "Visual Explanation"
    )

    st.caption(
        "Three complementary views: the annotated lesion, "
        "model attention, and the lesion-focused model input."
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
                caption="Original + Ground-Truth Mask",
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
            caption="BBOX-Based Lesion Crop",
            use_container_width=True
        )

    st.caption(
        "The ground-truth mask is the reference lesion annotation. "
        "Grad-CAM highlights regions contributing to the model prediction."
    )


# ============================================================
# UPLOAD ANALYSIS
# ============================================================

else:

    st.header(
        "Analyze Your Ultrasound"
    )

    st.caption(
        "Upload a breast ultrasound image. "
        "The lesion-focused model view will be generated automatically."
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

        width, height = image.size

        st.image(
            image,
            caption="Uploaded Ultrasound",
            width=500
        )

        st.caption(
            f"Image size: {width} × {height} pixels"
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

                    # Automatic lesion-focused crop

                    crop_image = (
                        generate_automatic_crop(
                            model,
                            image
                        )
                    )

                    # Final dual-view prediction

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

                    # Final Grad-CAM

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

                # ------------------------------------------------
                # RESULT
                # ------------------------------------------------

                st.subheader(
                    "AI Prediction"
                )

                if prediction == "Malignant":

                    st.error(
                        f"Prediction: {prediction}"
                    )

                else:

                    st.success(
                        f"Prediction: {prediction}"
                    )

                p1, p2 = st.columns(2)

                with p1:

                    st.metric(
                        "Benign probability",
                        f"{benign_probability * 100:.1f}%"
                    )

                with p2:

                    st.metric(
                        "Malignant probability",
                        f"{malignant_probability * 100:.1f}%"
                    )

                # ------------------------------------------------
                # THREE VISUALS
                # ------------------------------------------------

                st.subheader(
                    "Visual Explanation"
                )

                st.caption(
                    "The lesion-focused view is automatically "
                    "generated from model attention."
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
                    "No ground-truth mask is displayed for uploaded "
                    "images because reference lesion annotations are "
                    "not available."
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
    "Model Information"
):

    st.write(
        f"**Architecture:** Dual EfficientNet-B3"
    )

    st.write(
        "**Task:** Benign vs Malignant"
    )

    st.write(
        f"**Input resolution:** {IMAGE_SIZE} × {IMAGE_SIZE}"
    )

    st.write(
        f"**Decision threshold:** {THRESHOLD}"
    )

    st.write(
        f"**Device:** {device}"
    )

    st.write(
        "**Explainability:** Grad-CAM"
    )

    st.write(
        "**Sample lesion localization:** BUS-BRA BBOX"
    )

    st.write(
        "**Uploaded-image localization:** AI-generated attention crop"
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="disclaimer">

    <strong>Research & Demonstration Prototype</strong><br>

    This prototype is for research and educational use only,
    not clinical diagnosis. Performance may improve with larger,
    more diverse datasets and further model training.

    </div>
    """,
    unsafe_allow_html=True
)

st.caption(
    "SonoInsight AI · Breast Ultrasound Research Prototype"
)
