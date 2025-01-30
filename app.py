import streamlit as st
from PIL import Image
import os
from pathlib import Path
import google.generativeai as genai

# Configure the Google Gemini API key
GOOGLE_API_KEY = "AIzaSyDpRJ5rOYh4tNIJcnntNS6K3xWuswval6c"
genai.configure(api_key=GOOGLE_API_KEY)

# Model Configuration
MODEL_CONFIG = {
    "temperature": 0.2,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

# Safety Settings
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# Load the Gemini model with configurations
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash", 
    generation_config=MODEL_CONFIG, 
    safety_settings=safety_settings
)

# Define image format to input in Gemini
def image_format(image_path):
    img = Path(image_path)
    if not img.exists():
        raise FileNotFoundError(f"Could not find image: {img}")
    image_parts = [{"mime_type": "image/png", "data": img.read_bytes()}]
    return image_parts

# Gemini pro model output
def gemini_output(image_path, system_prompt, user_prompt):
    try:
        image_info = image_format(image_path)
        input_prompt = [system_prompt, image_info[0], user_prompt]
        response = model.generate_content(input_prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# Apply custom CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #f0f4f7;
            color: #333333;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .stTextInput > div > div > input {
            border: 2px solid #0366fc;
            border-radius: 8px;
            padding: 10px;
        }
        .stButton > button {
            background-color: #0366fc;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 8px 16px;
            font-size: 16px;
            cursor: pointer;
        }
        .stButton > button:hover {
            background-color: #024db6;
        }
        .stImage {
            margin-bottom: 20px;
            border: 2px solid #ccc;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit app
st.title("🌿 Disease Prediction App")
st.markdown("### Analyze disease images with the power of AI! 🌟")
st.write("Upload an image and provide a custom prompt for extracting disease-related information")

# Image input
image_input = st.file_uploader("📤 Upload an Image", type=["png", "jpg", "jpeg", "webp"])

# Display the uploaded image
if image_input is not None:
    st.image(image_input, caption="📷 Uploaded Image", use_container_width=True)

# User prompt input
user_prompt_input = st.text_input("💡 Enter a Custom Prompt", placeholder="E.g., 'Analyze for skin rashes in JSON format'")

# Predict button
if st.button("🔍 Predict Disease"):
    if image_input is not None and user_prompt_input:
        try:
            # Save the uploaded image
            image_path = os.path.join("/tmp/uploads", "uploaded_image.png")
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            with open(image_path, "wb") as f:
                f.write(image_input.getbuffer())

            # System prompt
            system_prompt = """
                Please upload an image, and I will analyze it for visible symptoms related to potential diseases, 
                such as skin rashes, lesions, or discoloration, in 30 words or less.
            """

            # Call the gemini_output function
            prediction = gemini_output(image_path, system_prompt, user_prompt_input)
            st.success(f"🔍 Prediction: {prediction}")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    else:
        st.warning("⚠ Please upload an image and provide a prompt.")