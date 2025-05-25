import streamlit as st
import streamlit as st
import subprocess
import os
from PIL import Image
import tempfile
import re
import time

st.set_page_config(page_title="OCR Text Recognition", layout="wide")

st.title("OCR Text Recognition using PP-OCRv4")

# Function to run OCR inference
def run_ocr_inference(image_path):
    # Change to PaddleOCR directory
    paddle_ocr_dir = r"C:\Users\15104\Desktop\MLfinal\data\PaddleOCR-release-2.8"
    original_dir = os.getcwd()
    
    try:
        os.chdir(paddle_ocr_dir)
        cmd = [
            "python",
            "tools/infer_rec.py",
            "-c", "configs/rec/PP-OCRv4/en_PP-OCRv4_rec.yml",
            "-o", f"Global.pretrained_model=pretrain_models/en_PP-OCRv4_rec_train/best_accuracy",
            f"Global.infer_img={image_path}"
        ]
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        end_time = time.time()
        elapsed = end_time - start_time
        if result.stdout:
            for line in result.stdout.split('\n'):
                if "result:" in line:
                    match = re.search(r'result: (.*?)\t(\d+\.\d+)', line)
                    if match:
                        text = match.group(1)
                        confidence = float(match.group(2))
                        return f"Recognized Text: {text}\nConfidence: {confidence:.2%}\nRun time: {elapsed:.2f}s"
        if result.stderr:
            st.error("Error occurred during OCR processing")
            return None
        return None
    except Exception as e:
        st.error(f"Error running OCR: {str(e)}")
        return None
    finally:
        os.chdir(original_dir)

st.write("### Upload an image for OCR recognition")

uploaded_file = st.file_uploader("Upload your image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_path = tmp_file.name
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    if st.button("Run OCR"):
        with st.spinner("Running OCR inference..."):
            result = run_ocr_inference(temp_path)
            if result:
                st.write("### OCR Results")
                st.markdown(f"```\n{result}\n```")
    os.unlink(temp_path)
