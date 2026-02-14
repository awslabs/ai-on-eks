from PyPDF2 import PdfReader

from pathlib import Path
import logging, base64


def encode_image(image_path):
    """Encode image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
# Add this function to handle PDF processing
def process_pdf(pdf_path: str) -> str:
    """Process PDF and return its content"""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        logging.error(f"PDF processing error: {str(e)}")
        return ""