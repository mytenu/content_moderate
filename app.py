import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64
from datetime import datetime


# Set page config
st.set_page_config(page_title="Sticker Moderation Chat", layout="centered")
st.title("🛡️ Sticker Moderation Chat")



# Load the CNN model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cnn_model.h5")

model = load_model()

# CIFAR-10 class labels
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Function to preprocess image
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((32, 32))
    image = image.convert('RGB')
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, 32, 32, 3)
    return img_array

# Convert image to base64
def image_to_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Convert base64 back to image
def base64_to_image(b64_string: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_string)))

# Format timestamp
def timestamp() -> str:
    return datetime.now().strftime("%H:%M")



# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for entry in st.session_state.chat_history:
    if entry["type"] == "text":
        with st.container():
            st.markdown(f"""
            <div style="background-color:#f0f2f6; padding:10px; border-radius:10px; margin-bottom:5px;">
                <strong>🧑 You</strong> <span style="color:gray; font-size: 10px;">{entry['time']}</span><br>
                {entry['data']}
            </div>
            """, unsafe_allow_html=True)

    elif entry["type"] == "sticker":
        with st.container():
            st.markdown(f"""
            <div style="background-color:#e0f7e9; padding:10px; border-radius:10px; margin-bottom:5px;">
                <strong>🧑 You</strong> <span style="color:gray; font-size: 10px;">{entry['time']}</span><br>
                <span>🖼️ Sticker:</span>
            </div>
            """, unsafe_allow_html=True)
            st.image(base64_to_image(entry["data"]), width=120)

    elif entry["type"] == "warning":
        with st.container():
            st.markdown(f"""
            <div style="background-color:#ffe6e6; padding:10px; border-radius:10px; margin-bottom:5px;">
                <strong>🚫 System</strong> <span style="color:gray; font-size: 10px;">{entry['time']}</span><br>
                {entry['data']}
            </div>
            """, unsafe_allow_html=True)

# Input section
col1, col2 = st.columns([6, 1])
with col1:
    user_input = st.text_input("Send a message", label_visibility="collapsed")
with col2:
    send_text = st.button("Send Text")

# Handle text input
if send_text and user_input.strip() != "":
    st.session_state.chat_history.append({
        "type": "text",
        "data": user_input.strip(),
        "time": timestamp()
    })

# Sticker section
sticker_col, button_col = st.columns([6, 1])
with sticker_col:
    sticker = st.file_uploader("Send Sticker", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
with button_col:
    send_sticker = st.button("Send Sticker")

# Handle sticker input
if send_sticker and sticker is not None:
    img = Image.open(sticker)
    st.image(img, caption="Sticker uploaded", width=100)

    processed = preprocess_image(img)
    preds = model.predict(processed)
    pred_class = class_names[np.argmax(preds)]

    st.write(f"🧠 Model Prediction: **{pred_class}**")

    if pred_class == "horse":
        st.session_state.chat_history.append({
            "type": "warning",
            "data": "This sticker flouts the rules of healthy conversation and has been flagged for deletion.",
            "time": timestamp()
        })
    else:
        img_b64 = image_to_base64(img)
        st.session_state.chat_history.append({
            "type": "sticker",
            "data": img_b64,
            "time": timestamp()
        })
