import streamlit as st
from PIL import Image, UnidentifiedImageError
from transformers import pipeline
from io import BytesIO
import tempfile
import os

st.set_page_config(
    page_title="Deepfake Detection AI",
    page_icon="🔍",
    layout="wide"
)

MODEL_NAME = "Wvolf/ViT_Deepfake_Detection"


@st.cache_resource(show_spinner="Loading pretrained AI model...")
def load_model():
    return pipeline(
        "image-classification",
        model=MODEL_NAME,
        device=-1
    )


def classify_image(image, model):
    image = image.convert("RGB")
    predictions = model(image, top_k=None)

    scores = {
        item["label"].strip().lower(): float(item["score"])
        for item in predictions
    }

    real = next(
        (score for label, score in scores.items() if "real" in label),
        None
    )
    fake = next(
        (score for label, score in scores.items() if "fake" in label),
        None
    )

    if real is None or fake is None:
        raise ValueError(
            f"Unexpected model labels: {list(scores.keys())}"
        )

    return ("FAKE" if fake > real else "REAL"), real, fake


def display_result(label, real, fake):
    a, b, c = st.columns(3)
    a.metric("Prediction", label)
    b.metric("Real score", f"{real * 100:.2f}%")
    c.metric("Fake score", f"{fake * 100:.2f}%")

    if label == "FAKE":
        st.error("The model predicts this content is FAKE.")
    else:
        st.success("The model predicts this content is REAL.")

    st.caption(
        "This is an AI estimate, not proof of authenticity."
    )


st.title("Deepfake Image and Video Detection")
st.subheader(
    "A Deep Learning-Based Framework for Robust Detection "
    "of Deepfake Images and Video Content"
)

st.info(
    "Upload an image or video for a preliminary AI analysis. "
    "The model may make incorrect predictions."
)

image_tab, video_tab, about_tab = st.tabs(
    ["Image Detection", "Video Detection", "About"]
)

with image_tab:
    st.header("Upload Images")

    files = st.file_uploader(
        "Select JPG, JPEG, PNG, or WEBP images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True
    )

    if files:
        for file in files:
            try:
                image = Image.open(
                    BytesIO(file.getvalue())
                ).convert("RGB")
                st.image(
                    image,
                    caption=file.name,
                    use_container_width=True
                )
            except (UnidentifiedImageError, OSError, ValueError):
                st.error(f"Cannot read image: {file.name}")

        if st.button("Detect Real or Fake", type="primary"):
            try:
                model = load_model()

                for file in files:
                    try:
                        image = Image.open(
                            BytesIO(file.getvalue())
                        ).convert("RGB")

                        label, real, fake = classify_image(image, model)

                        st.divider()
                        st.write(f"**File:** {file.name}")
                        display_result(label, real, fake)

                    except Exception as exc:
                        st.error(f"Image analysis failed: {exc}")

            except Exception as exc:
                st.error("Could not load the pretrained model.")
                st.code(str(exc))

with video_tab:
    st.header("Upload Video")

    video = st.file_uploader(
        "Select MP4, MOV, AVI, or WEBM",
        type=["mp4", "mov", "avi", "webm", "m4v"],
        key="video_upload"
    )

    number_of_frames = st.slider(
        "Frames to analyze",
        min_value=3,
        max_value=8,
        value=4
    )

    if video:
        video_bytes = video.getvalue()
        st.video(video_bytes)

        if st.button("Analyze Video", type="primary"):
            path = None
            reader = None

            try:
                import imageio.v2 as imageio

                model = load_model()

                suffix = os.path.splitext(video.name)[1] or ".mp4"

                with tempfile.NamedTemporaryFile(
                    suffix=suffix, delete=False
                ) as temp:
                    temp.write(video_bytes)
                    path = temp.name

                with st.spinner("Analyzing video frames..."):
                    reader = imageio.get_reader(
                        path, format="FFMPEG"
                    )
                    meta = reader.get_meta_data()
                    total = meta.get("nframes")

                    if not isinstance(total, int) or total <= 0 or total > 10**7:
                        total = reader.count_frames()

                    if total <= 0:
                        raise ValueError("No readable video frames.")

                    count = min(number_of_frames, total)

                    if count == 1:
                        indexes = [0]
                    else:
                        indexes = [
                            round(i * (total - 1) / (count - 1))
                            for i in range(count)
                        ]

                    results = []

                    for index in indexes:
                        frame = Image.fromarray(
                            reader.get_data(index)
                        ).convert("RGB")

                        label, real, fake = classify_image(frame, model)

                        results.append({
                            "index": index,
                            "image": frame,
                            "label": label,
                            "real": real,
                            "fake": fake
                        })

                avg_real = sum(x["real"] for x in results) / len(results)
                avg_fake = sum(x["fake"] for x in results) / len(results)
                overall = "FAKE" if avg_fake > avg_real else "REAL"

                st.subheader("Video Prediction")
                display_result(overall, avg_real, avg_fake)

                st.warning(
                    "This result averages sampled frame scores. "
                    "It does not analyze motion or temporal inconsistencies."
                )

                for result in results:
                    with st.expander(
                        f"Frame {result['index']}: {result['label']}"
                    ):
                        st.image(
                            result["image"],
                            use_container_width=True
                        )
                        st.write(
                            f"Real: {result['real'] * 100:.2f}%"
                        )
                        st.write(
                            f"Fake: {result['fake'] * 100:.2f}%"
                        )

            except Exception as exc:
                st.error("Video analysis failed.")
                st.code(str(exc))

            finally:
                if reader is not None:
                    try:
                        reader.close()
                    except Exception:
                        pass

                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

with about_tab:
    st.header("Project Information")
    st.markdown("""
    **Objectives**
    1. Upload images and videos.
    2. Apply a pretrained image classifier.
    3. Display preliminary REAL/FAKE estimates.

    **Limitations**
    - Results depend on the pretrained model and input quality.
    - The model can misclassify genuine or manipulated content.
    - Video predictions are based on sampled frames, not a dedicated
      temporal video model.
    - Test the model on an independent dataset before reporting accuracy.
    """)

st.divider()
st.caption("Deepfake Detection AI | Academic Prototype")
