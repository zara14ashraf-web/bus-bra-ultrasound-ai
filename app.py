import os
import json
import requests

import torch
import streamlit as st

from PIL import Image
from torchvision import transforms

from model import SharedDualEfficientNetB3


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="BUS-BRA Breast Ultrasound AI",
    page_icon="🩺",
    layout="centered"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

CONFIG_PATH = os.path.join(
    BASE_DIR,
    "deployment_config.json"
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
# LOAD CONFIGURATION
# ============================================================

with open(
    CONFIG_PATH,
    "r"
) as f:

    config = json.load(f)


MODEL_NAME = config["model_name"]
CLASSES = config["classes"]

IMAGE_SIZE = config["input_size"]
THRESHOLD = config["threshold"]
CROP_MARGIN = config["crop_margin"]

MEAN = config["normalization"]["mean"]
STD = config["normalization"]["std"]


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

    if os.path.exists(
        CHECKPOINT_PATH
    ):
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

    if isinstance(
        checkpoint,
        dict
    ):

        if "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        elif "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model = model.to(
        device
    )

    model.eval()

    return model


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([
    transforms.Resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
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
    box,
    margin=0.25
):

    x1, y1, x2, y2 = [
        int(v)
        for v in box
    ]

    width, height = image.size

    box_width = x2 - x1
    box_height = y2 - y1

    pad_x = int(
        box_width * margin
    )

    pad_y = int(
        box_height * margin
    )

    x1 = max(
        0,
        x1 - pad_x
    )

    y1 = max(
        0,
        y1 - pad_y
    )

    x2 = min(
        width,
        x2 + pad_x
    )

    y2 = min(
        height,
        y2 + pad_y
    )

    return image.crop(
        (
            x1,
            y1,
            x2,
            y2
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
    ).unsqueeze(0)

    crop_image = make_lesion_crop(
        image,
        lesion_box,
        CROP_MARGIN
    )

    crop_tensor = transform(
        crop_image
    ).unsqueeze(0)

    full_tensor = full_tensor.to(
        device
    )

    crop_tensor = crop_tensor.to(
        device
    )

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

    if malignant_probability >= THRESHOLD:

        prediction = "Malignant"

    else:

        prediction = "Benign"

    return {
        "prediction": prediction,
        "benign_probability":
            benign_probability,
        "malignant_probability":
            malignant_probability,
        "threshold": THRESHOLD
    }, crop_image


# ============================================================
# HEADER
# ============================================================

st.title(
    "🩺 BUS-BRA Breast Ultrasound AI"
)

st.write(
    "Dual-view EfficientNet-B3 model "
    "for benign vs malignant breast "
    "ultrasound classification."
)


st.info(
    "Research prototype — this tool is "
    "not a clinical diagnostic system."
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
# IMAGE UPLOAD
# ============================================================

st.subheader(
    "1. Upload Ultrasound Image"
)

uploaded_file = st.file_uploader(
    "Choose a breast ultrasound image",
    type=[
        "png",
        "jpg",
        "jpeg"
    ]
)


# ============================================================
# IMAGE PROCESSING
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded ultrasound",
        use_container_width=True
    )

    width, height = image.size

    st.caption(
        f"Image size: {width} × {height} pixels"
    )

    # --------------------------------------------------------
    # LESION BOX
    # --------------------------------------------------------

    st.subheader(
        "2. Define Lesion Region"
    )

    st.write(
        "Enter the lesion bounding-box "
        "coordinates in the original image."
    )

    col1, col2 = st.columns(2)

    with col1:

        x1 = st.number_input(
            "X1",
            min_value=0,
            max_value=width,
            value=0,
            step=1
        )

        y1 = st.number_input(
            "Y1",
            min_value=0,
            max_value=height,
            value=0,
            step=1
        )

    with col2:

        x2 = st.number_input(
            "X2",
            min_value=0,
            max_value=width,
            value=width,
            step=1
        )

        y2 = st.number_input(
            "Y2",
            min_value=0,
            max_value=height,
            value=height,
            step=1
        )

    lesion_box = [
        x1,
        y1,
        x2,
        y2
    ]

    # --------------------------------------------------------
    # VALIDATE BOX
    # --------------------------------------------------------

    valid_box = (
        x2 > x1
        and y2 > y1
    )

    if not valid_box:

        st.warning(
            "Please provide a valid lesion box."
        )

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    if valid_box:

        if st.button(
            "🔍 Analyze Ultrasound",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing ultrasound..."
            ):

                try:

                    result, crop_image = predict(
                        model,
                        image,
                        lesion_box
                    )

                    prediction = result[
                        "prediction"
                    ]

                    benign_probability = result[
                        "benign_probability"
                    ]

                    malignant_probability = result[
                        "malignant_probability"
                    ]

                    # ----------------------------------------
                    # RESULTS
                    # ----------------------------------------

                    st.subheader(
                        "3. AI Prediction"
                    )

                    if prediction == "Malignant":

                        st.error(
                            f"Prediction: {prediction}"
                        )

                    else:

                        st.success(
                            f"Prediction: {prediction}"
                        )

                    # ----------------------------------------
                    # PROBABILITIES
                    # ----------------------------------------

                    col1, col2 = st.columns(2)

                    with col1:

                        st.metric(
                            "Benign probability",
                            f"{benign_probability * 100:.2f}%"
                        )

                    with col2:

                        st.metric(
                            "Malignant probability",
                            f"{malignant_probability * 100:.2f}%"
                        )

                    # ----------------------------------------
                    # LESION CROP
                    # ----------------------------------------

                    st.subheader(
                        "Lesion-Focused View"
                    )

                    st.image(
                        crop_image,
                        caption="Lesion-focused crop",
                        use_container_width=True
                    )

                    # ----------------------------------------
                    # MODEL INFORMATION
                    # ----------------------------------------

                    with st.expander(
                        "Model information"
                    ):

                        st.write(
                            f"**Model:** {MODEL_NAME}"
                        )

                        st.write(
                            "**Architecture:** "
                            "Dual EfficientNet-B3"
                        )

                        st.write(
                            "**Input size:** "
                            f"{IMAGE_SIZE} × {IMAGE_SIZE}"
                        )

                        st.write(
                            "**Decision threshold:** "
                            f"{THRESHOLD}"
                        )

                        st.write(
                            "**Device:** "
                            f"{device}"
                        )

                except Exception as e:

                    st.error(
                        "Prediction failed."
                    )

                    st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "BUS-BRA Dual-View Breast Ultrasound AI "
    "Research Prototype"
)

