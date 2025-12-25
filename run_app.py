import streamlit as st
import sys
import os

# Add backend folder to sys.path so we can import server.py
sys.path.append(os.path.join(os.getcwd(), "backend"))

# Import everything from your working backend
import server

st.set_page_config(page_title="Alzheimer's Voice Detection", layout="wide")

st.title("Alzheimer's Voice Detection")
st.write("This mirrors the Emergent preview!")

# Provide file uploader
uploaded_file = st.file_uploader("Upload a recording", type=["wav", "mp3"])
if uploaded_file is not None:
    # Call your backend function for prediction
    try:
        # Replace 'predict' with the exact function in server.py that does prediction
        result = server.predict(uploaded_file)
        st.success(f"Prediction: {result}")
    except Exception as e:
        st.error(f"Error: {e}")
