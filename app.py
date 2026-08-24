```python
import os
import json
import torch
import streamlit as st
from PIL import Image
from torchvision import transforms
from model import SharedDualEfficientNetB3


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="BUS-BRA Ultrasound AI",
    page_icon="🩺",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHECKPOINT_PATH = os.path.join(
    BASE_DIR,
    "best_dual_effnet_b3.pth"
)

CONFIG_PATH = os.path.join(
    BASE_DIR,
    "deployment_config.json"
)


# ============================================================
# LOAD CONFIGURATION
# ============================================================

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

IMAGE_SIZE = config["input_size"]
THRESHOLD = config["threshold"]
CROP_MARGIN = config["crop_margin"]

MEAN = config["normalization"]["mean"]
STD = config["normalization"]["std"]

CLASSES = config["classes"]


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():

    model = SharedDualEfficientNetB3(
        num_classes=len(CLASSES)
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
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

    model = model.to(DEVICE)
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
    box,
    margin=0.25
):

    x1, y1, x2, y2 = [
        int(v) for v in box
    ]

    width, height = image.size

    box_w = x2 - x1
    box_h = y2 - y1

    pad_x = int(box_w * margin)
    pad_y = int(box_h * margin)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)

    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)

    return image.crop(
        (x1, y1, x2, y2)
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

    full_tensor = full_tensor.to(DEVICE)
    crop_tensor = crop_tensor.to(DEVICE)

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

    confidence = (
        malignant_probability
        if prediction == "Malignant"
        else benign_probability
    )

    return {
        "prediction": prediction,
        "confidence": confidence,
        "benign_probability": benign_probability,
        "malignant_probability": malignant_probability,
        "crop_image": crop_image
    }


# ============================================================
# HEADER
# ============================================================

st.title("🩺 BUS-BRA Ultrasound AI")

st.markdown(
    """
### Dual-View Breast Ultrasound Classification

An AI-assisted research prototype using a **dual-view
EfficientNet-B3 architecture** that analyzes both the complete
ultrasound image and a lesion-focused region.
"""
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("About the Model")

    st.write(
        "The model uses two EfficientNet-B3 branches:"
    )

    st.write(
        "• Full ultrasound image"
    )

    st.write(
        "• Lesion-focused crop"
    )

    st.write(
        "Their feature representations are combined "
        "for binary classification."
    )

    st.divider()

    st.caption(
        f"Input size: {IMAGE_SIZE} × {IMAGE_SIZE}"
    )

    st.caption(
        f"Decision threshold: {THRESHOLD:.2f}"
    )

    st.caption(
        "Research prototype — not a clinical diagnostic tool."
    )


# ============================================================
# MODEL
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

uploaded_file = st.file_uploader(
    "Upload a breast ultrasound image",
    type=[
        "png",
        "jpg",
        "jpeg",
        "bmp",
        "tif",
        "tiff"
    ]
)


if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.subheader("1. Ultrasound Image")

    st.image(
        image,
        caption="Uploaded ultrasound",
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # LESION COORDINATES
    # --------------------------------------------------------

    st.subheader("2. Define Lesion Region")

    st.info(
        "Enter the lesion bounding-box coordinates. "
        "The coordinates correspond to the original image."
    )

    width, height = image.size

    col1, col2 = st.columns(2)

    with col1:

        x1 = st.number_input(
            "Left (x1)",
            min_value=0,
            max_value=width,
            value=0,
            step=1
        )

        y1 = st.number_input(
            "Top (y1)",
            min_value=0,
            max_value=height,
            value=0,
            step=1
        )

    with col2:

        x2 = st.number_input(
            "Right (x2)",
            min_value=0,
            max_value=width,
            value=width,
            step=1
        )

        y2 = st.number_input(
            "Bottom (y2)",
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

    valid_box = (
        x2 > x1
        and y2 > y1
    )

    if valid_box:

        crop_preview = make_lesion_crop(
            image,
            lesion_box,
            CROP_MARGIN
        )

        st.subheader("Lesion-Focused View")

        preview_col1, preview_col2 = st.columns(2)

        with preview_col1:

            st.image(
                image,
                caption="Full ultrasound",
                use_container_width=True
            )

        with preview_col2:

            st.image(
                crop_preview,
                caption="Lesion-focused crop",
                use_container_width=True
            )

        st.divider()

        # ----------------------------------------------------
        # RUN INFERENCE
        # ----------------------------------------------------

        if st.button(
            "🔬 Analyze Ultrasound",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing ultrasound..."
            ):

                result = predict(
                    model,
                    image,
                    lesion_box
                )

            st.subheader(
                "3. AI Assessment"
            )

            prediction = result[
                "prediction"
            ]

            confidence = result[
                "confidence"
            ]

            benign_probability = result[
                "benign_probability"
            ]

            malignant_probability = result[
                "malignant_probability"
            ]

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            if prediction == "Malignant":

                st.error(
                    f"Prediction: {prediction}"
                )

            else:

                st.success(
                    f"Prediction: {prediction}"
                )

            st.metric(
                "Model confidence",
                f"{confidence * 100:.2f}%"
            )

            st.divider()

            # ------------------------------------------------
            # PROBABILITIES
            # ------------------------------------------------

            st.subheader(
                "Class Probabilities"
            )

            prob_col1, prob_col2 = st.columns(2)

            with prob_col1:

                st.metric(
                    "Benign",
                    f"{benign_probability * 100:.2f}%"
                )

            with prob_col2:

                st.metric(
                    "Malignant",
                    f"{malignant_probability * 100:.2f}%"
                )

            st.progress(
                benign_probability
            )

            st.caption(
                "Benign probability"
            )

            st.progress(
                malignant_probability
            )

            st.caption(
                "Malignant probability"
            )

            st.divider()

            # ------------------------------------------------
            # MODEL DETAILS
            # ------------------------------------------------

            st.subheader(
                "Model Interpretation"
            )

            st.write(
                """
The prediction is generated from two complementary views:
the complete ultrasound image and the lesion-focused crop.
Both views are processed independently by EfficientNet-B3
feature extractors, after which their learned representations
are combined by the classification head.
"""
            )

            st.warning(
                "This system is a research prototype and should "
                "not be used as a substitute for professional "
                "radiological assessment or clinical diagnosis."
            )

    else:

        st.warning(
            "Please provide a valid lesion region where "
            "x2 > x1 and y2 > y1."
        )

else:

    st.info(
        "Upload a breast ultrasound image to begin."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "BUS-BRA Ultrasound AI • Research Prototype"
)
```
