import os
import requests

import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st

from PIL import Image, ImageDraw
from torchvision import transforms

from model import SharedDualEfficientNetB3


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="BUS-BRA Breast Ultrasound AI",
    page_icon="🩺",
    layout="wide"
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
# BUS-BRA SAMPLE CASES
# ============================================================

SAMPLES = {

    "Sample 1 — Benign": {
        "image": "sample_1_bus_0002-l.png",
        "mask": "sample_1_bus_0002-l_MASK.png",
        "bbox": [134, 142, 88, 50],
        "label": "Benign",
        "histology": "fibroadenoma"
    },

    "Sample 2 — Benign": {
        "image": "sample_2_bus_0002-r.png",
        "mask": "sample_2_bus_0002-r_MASK.png",
        "bbox": [113, 143, 68, 47],
        "label": "Benign",
        "histology": "fibroadenoma"
    },

    "Sample 3 — Malignant": {
        "image": "sample_3_bus_0001-l.png",
        "mask": "sample_3_bus_0001-l_MASK.png",
        "bbox": [91, 24, 103, 79],
        "label": "Malignant",
        "histology": "invasive ductal carcinoma"
    },

    "Sample 4 — Malignant": {
        "image": "sample_4_bus_0001-r.png",
        "mask": "sample_4_bus_0001-r_MASK.png",
        "bbox": [102, 24, 82, 79],
        "label": "Malignant",
        "histology": "invasive ductal carcinoma"
    }
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
        "Downloading trained AI model..."
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
        ) as f:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):

                if chunk:
                    f.write(chunk)


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
    )
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
# DRAW BOUNDING BOX
# ============================================================

def draw_bbox(
    image,
    bbox
):

    output = image.copy()

    draw = ImageDraw.Draw(output)

    x, y, width, height = [
        int(v)
        for v in bbox
    ]

    draw.rectangle(
        [
            x,
            y,
            x + width,
            y + height
        ],
        outline="red",
        width=3
    )

    return output


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

    mask_array = np.array(mask)

    mask_binary = mask_array > 0

    if not mask_binary.any():

        return image

    output = image_array.copy()

    output[mask_binary, 0] = 255
    output[mask_binary, 1] *= 0.35
    output[mask_binary, 2] *= 0.35

    output = (
        0.65 * image_array
        + 0.35 * output
    )

    output = np.clip(
        output,
        0,
        255
    ).astype(
        np.uint8
    )

    return Image.fromarray(output)


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
        self.target_layer = target_layer

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

        cam = cam.detach().cpu().numpy()

        cam -= cam.min()

        if cam.max() > 0:

            cam /= cam.max()

        return cam

    def remove_hooks(self):

        self.forward_hook.remove()
        self.backward_hook.remove()


# ============================================================
# GRAD-CAM OVERLAY
# ============================================================

def create_gradcam_overlay(
    image,
    cam
):

    image_resized = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
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
    lesion_box
):

    full_tensor = transform(
        image
    ).unsqueeze(
        0
    ).to(device)

    crop_image = make_lesion_crop(
        image,
        lesion_box
    )

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

    return {
        "prediction": prediction,
        "benign_probability": benign_probability,
        "malignant_probability": malignant_probability,
        "full_tensor": full_tensor,
        "crop_tensor": crop_tensor
    }, crop_image


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
# HEADER
# ============================================================

st.title(
    "🩺 BUS-BRA Breast Ultrasound AI"
)

st.write(
    "Dual-view EfficientNet-B3 for "
    "benign vs malignant breast "
    "ultrasound classification."
)

st.info(
    "Research prototype — not a clinical "
    "diagnostic system."
)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = load_model()

except Exception as e:

    st.error(
        "Unable to load the trained model."
    )

    st.exception(e)

    st.stop()


# ============================================================
# SAMPLE CASES
# ============================================================

st.header(
    "🔬 BUS-BRA Sample Cases"
)

st.write(
    "Select a BUS-BRA case to view the "
    "AI prediction, lesion localization, "
    "ground-truth mask and Grad-CAM."
)

selected_sample = st.selectbox(
    "Select sample",
    list(SAMPLES.keys())
)

sample = SAMPLES[selected_sample]

sample_image_path = os.path.join(
    BASE_DIR,
    sample["image"]
)

sample_mask_path = os.path.join(
    BASE_DIR,
    sample["mask"]
)


# ============================================================
# LOAD SAMPLE IMAGE
# ============================================================

if not os.path.exists(sample_image_path):

    st.error(
        f"Sample image not found: "
        f"{sample['image']}"
    )

    st.stop()

sample_image = Image.open(
    sample_image_path
).convert("RGB")


# ============================================================
# LOAD SAMPLE MASK
# ============================================================

sample_mask = None

if os.path.exists(sample_mask_path):

    sample_mask = Image.open(
        sample_mask_path
    ).convert("L")


# ============================================================
# SAMPLE ANALYSIS
# ============================================================

bbox = sample["bbox"]

try:

    result, crop_image = predict(
        model,
        sample_image,
        bbox
    )

    prediction = result["prediction"]

    benign_probability = (
        result["benign_probability"]
    )

    malignant_probability = (
        result["malignant_probability"]
    )

    full_tensor = result["full_tensor"]

    crop_tensor = result["crop_tensor"]

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

    gradcam_overlay = (
        create_gradcam_overlay(
            sample_image,
            cam
        )
    )

    bbox_image = draw_bbox(
        sample_image,
        bbox
    )

except Exception as e:

    st.error(
        "Sample analysis failed."
    )

    st.exception(e)

    st.stop()


# ============================================================
# CASE INFORMATION
# ============================================================

st.subheader(
    "Case Information"
)

info1, info2, info3 = st.columns(3)

with info1:

    st.write(
        f"**Pathology:** {sample['label']}"
    )

with info2:

    st.write(
        f"**Histology:** {sample['histology']}"
    )

with info3:

    st.write(
        f"**BBOX:** {bbox}"
    )


# ============================================================
# PREDICTION
# ============================================================

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
        f"{benign_probability * 100:.2f}%"
    )

with p2:

    st.metric(
        "Malignant probability",
        f"{malignant_probability * 100:.2f}%"
    )


# ============================================================
# VISUAL EXPLANATION
# ============================================================

st.subheader(
    "Visual Explanation"
)

v1, v2 = st.columns(2)

with v1:

    st.image(
        sample_image,
        caption="Original ultrasound",
        width=280
    )

with v2:

    st.image(
        bbox_image,
        caption="Lesion bounding box",
        width=280
    )


v3, v4 = st.columns(2)

with v3:

    if sample_mask is not None:

        mask_overlay = create_mask_overlay(
            sample_image,
            sample_mask
        )

        st.image(
            mask_overlay,
            caption="Ground-truth lesion mask",
            width=280
        )

    else:

        st.warning(
            "Ground-truth mask unavailable."
        )

with v4:

    st.image(
        gradcam_overlay,
        caption="Grad-CAM model attention",
        width=280
    )


# ============================================================
# LESION-FOCUSED CROP
# ============================================================

st.subheader(
    "Lesion-Focused Model Input"
)

st.image(
    crop_image,
    caption="BBOX-based lesion crop",
    width=280
)

st.caption(
    "Grad-CAM shows regions that contributed "
    "to the selected prediction. The ground-truth "
    "mask shows the annotated lesion region."
)


# ============================================================
# UPLOAD YOUR OWN IMAGE
# ============================================================

st.divider()

st.header(
    "📤 Analyze Your Own Ultrasound"
)

uploaded_file = st.file_uploader(
    "Choose a breast ultrasound image",
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
        caption="Uploaded ultrasound",
        width=420
    )

    st.caption(
        f"Image size: {width} × {height} pixels"
    )

    st.subheader(
        "Define Lesion Bounding Box"
    )

    st.write(
        "Enter X, Y, width and height "
        "of the lesion."
    )

    col1, col2 = st.columns(2)

    with col1:

        box_x = st.number_input(
            "X",
            min_value=0,
            max_value=max(0, width - 1),
            value=0,
            step=1
        )

        box_y = st.number_input(
            "Y",
            min_value=0,
            max_value=max(0, height - 1),
            value=0,
            step=1
        )

    with col2:

        box_width = st.number_input(
            "Lesion width",
            min_value=1,
            max_value=width,
            value=min(100, width),
            step=1
        )

        box_height = st.number_input(
            "Lesion height",
            min_value=1,
            max_value=height,
            value=min(100, height),
            step=1
        )

    lesion_box = [
        box_x,
        box_y,
        box_width,
        box_height
    ]

    valid_box = (
        box_x + box_width <= width
        and
        box_y + box_height <= height
    )

    if not valid_box:

        st.warning(
            "The lesion box extends outside "
            "the image. Please adjust it."
        )

    else:

        if st.button(
            "🔍 Analyze Ultrasound",
            type="primary",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "Analyzing ultrasound..."
                ):

                    result, upload_crop = predict(
                        model,
                        image,
                        lesion_box
                    )

                    upload_prediction = (
                        result["prediction"]
                    )

                    upload_benign = (
                        result["benign_probability"]
                    )

                    upload_malignant = (
                        result["malignant_probability"]
                    )

                    upload_full_tensor = (
                        result["full_tensor"]
                    )

                    upload_crop_tensor = (
                        result["crop_tensor"]
                    )

                    upload_class = (
                        1
                        if upload_prediction == "Malignant"
                        else 0
                    )

                    upload_cam = generate_gradcam(
                        model,
                        upload_full_tensor,
                        upload_crop_tensor,
                        upload_class
                    )

                    upload_gradcam = (
                        create_gradcam_overlay(
                            image,
                            upload_cam
                        )
                    )

                    upload_bbox = draw_bbox(
                        image,
                        lesion_box
                    )

                st.subheader(
                    "AI Prediction"
                )

                if upload_prediction == "Malignant":

                    st.error(
                        f"Prediction: "
                        f"{upload_prediction}"
                    )

                else:

                    st.success(
                        f"Prediction: "
                        f"{upload_prediction}"
                    )

                u1, u2 = st.columns(2)

                with u1:

                    st.metric(
                        "Benign probability",
                        f"{upload_benign * 100:.2f}%"
                    )

                with u2:

                    st.metric(
                        "Malignant probability",
                        f"{upload_malignant * 100:.2f}%"
                    )

                st.subheader(
                    "Explainability"
                )

                e1, e2 = st.columns(2)

                with e1:

                    st.image(
                        upload_bbox,
                        caption="Lesion bounding box",
                        width=320
                    )

                with e2:

                    st.image(
                        upload_gradcam,
                        caption="Grad-CAM attention",
                        width=320
                    )

                st.image(
                    upload_crop,
                    caption="Lesion-focused crop",
                    width=280
                )

                st.info(
                    "Ground-truth masks are shown for "
                    "the BUS-BRA sample cases because "
                    "their reference masks are available."
                )

            except Exception as e:

                st.error(
                    "Prediction failed."
                )

                st.exception(e)


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

with st.expander(
    "Model information"
):

    st.write(
        f"**Model:** {MODEL_NAME}"
    )

    st.write(
        "**Architecture:** Dual EfficientNet-B3"
    )

    st.write(
        f"**Input size:** {IMAGE_SIZE} × {IMAGE_SIZE}"
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
        "**Ground truth:** BUS-BRA lesion masks"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "BUS-BRA Dual-View Breast Ultrasound AI "
    "Research Prototype • Not for clinical diagnosis"
)
