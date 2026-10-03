import streamlit as st
import cv2
import numpy as np
from PIL import Image


# -------------------------------------------------
# Page Configuration
# -------------------------------------------------

st.set_page_config(
    page_title="OpenCV Image Filters",
    page_icon="🖼️",
    layout="wide"
)


# -------------------------------------------------
# Title
# -------------------------------------------------

st.title("🖼️ OpenCV Image Filters")

st.write(
    "Upload an image and apply different OpenCV filters."
)


# -------------------------------------------------
# Upload Image
# -------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"]
)


if uploaded_file is not None:

    # Read image
    image = Image.open(uploaded_file).convert("RGB")

    # Convert PIL image to NumPy
    image_np = np.array(image)

    # Convert RGB → BGR for OpenCV
    image_bgr = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2BGR
    )

    # -------------------------------------------------
    # Display Original Image
    # -------------------------------------------------

    st.subheader("Original Image")

    st.image(
        image,
        use_container_width=True
    )


    st.divider()

    st.subheader("Choose a Filter")


    # -------------------------------------------------
    # Five Buttons
    # -------------------------------------------------

    col1, col2, col3, col4, col5 = st.columns(5)


    # Default result
    result = image_bgr
    filter_name = "Original Image"


    # -------------------------------------------------
    # Button 1 - Grayscale
    # -------------------------------------------------

    with col1:

        if st.button(
            "⚫ Grayscale",
            use_container_width=True
        ):

            result = cv2.cvtColor(
                image_bgr,
                cv2.COLOR_BGR2GRAY
            )

            filter_name = "Grayscale"


    # -------------------------------------------------
    # Button 2 - Blur
    # -------------------------------------------------

    with col2:

        if st.button(
            "🌫️ Blur",
            use_container_width=True
        ):

            result = cv2.GaussianBlur(
                image_bgr,
                (15, 15),
                0
            )

            filter_name = "Gaussian Blur"


    # -------------------------------------------------
    # Button 3 - Edge Detection
    # -------------------------------------------------

    with col3:

        if st.button(
            "🔲 Edge",
            use_container_width=True
        ):

            gray = cv2.cvtColor(
                image_bgr,
                cv2.COLOR_BGR2GRAY
            )

            result = cv2.Canny(
                gray,
                100,
                200
            )

            filter_name = "Canny Edge Detection"


    # -------------------------------------------------
    # Button 4 - Image Enhancement
    # -------------------------------------------------

    with col4:

        if st.button(
            "✨ Enhance",
            use_container_width=True
        ):

            # Sharpening filter
            kernel = np.array([
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ])

            result = cv2.filter2D(
                image_bgr,
                -1,
                kernel
            )

            filter_name = "Sharpen"


    # -------------------------------------------------
    # Button 5 - Special Filters
    # -------------------------------------------------

    with col5:

        special_filter = st.selectbox(
            "🎨 Special",
            [
                "Choose...",
                "Sepia",
                "Negative",
                "Cartoon"
            ]
        )

        if special_filter == "Sepia":

            sepia_kernel = np.array([
                [0.272, 0.534, 0.131],
                [0.349, 0.686, 0.168],
                [0.393, 0.769, 0.189]
            ])

            result = cv2.transform(
                image_bgr,
                sepia_kernel
            )

            result = np.clip(
                result,
                0,
                255
            ).astype(np.uint8)

            filter_name = "Sepia"


        elif special_filter == "Negative":

            result = cv2.bitwise_not(
                image_bgr
            )

            filter_name = "Negative"


        elif special_filter == "Cartoon":

            gray = cv2.cvtColor(
                image_bgr,
                cv2.COLOR_BGR2GRAY
            )

            gray = cv2.medianBlur(
                gray,
                5
            )

            edges = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                9,
                9
            )

            color = cv2.bilateralFilter(
                image_bgr,
                9,
                250,
                250
            )

            result = cv2.bitwise_and(
                color,
                color,
                mask=edges
            )

            filter_name = "Cartoon"


    # -------------------------------------------------
    # Display Result
    # -------------------------------------------------

    st.divider()

    st.subheader(
        f"Result: {filter_name}"
    )


    # Convert OpenCV BGR → RGB
    if len(result.shape) == 2:

        result_rgb = result

        st.image(
            result_rgb,
            use_container_width=True
        )

    else:

        result_rgb = cv2.cvtColor(
            result,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            result_rgb,
            use_container_width=True
        )


else:

    st.info(
        "👆 Upload an image to start applying filters."
    )