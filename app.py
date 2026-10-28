"""
EduScan — AI-Powered Whiteboard Cleaner & Text Extractor

Flask backend:
1. Accepts a whiteboard image upload.
2. Creates an enhanced image for download.
3. Extracts text using PaddleOCR Small or qwen 2.5 VL model
4. Allows users to edit extracted text.
5. Exports the edited text as TXT or PDF.
"""

from pathlib import Path
from uuid import uuid4
from xml.sax.saxutils import escape

import base64
import secrets
import mimetypes
import os
import cv2
import json
import numpy as np
from image_processing import correct_perspective, detect_board_corners
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory,
    flash,
)
from werkzeug.utils import secure_filename
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------
# 1. Flask application configuration
# --------------------------------------------------

app = Flask(__name__)

# Secret key is used for Flask session messages.
# Replace this with an environment variable before deployment.
app.secret_key = os.getenv("FLASK_SECRET") or "dev-fallback-secret"

# Store uploaded images and generated files inside this folder.
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
OUTPUT_DIR = BASE_DIR / "static" / "outputs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Maximum upload size: 15 MB.
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


# --------------------------------------------------
# 2. PaddleOCR Small — load on first local-OCR fallback
# --------------------------------------------------

_MODEL_DIR = BASE_DIR / "models"
_DET_MODEL_DIR = _MODEL_DIR / "PP-OCRv6_small_det"
_REC_MODEL_DIR = _MODEL_DIR / "PP-OCRv6_small_rec"
_paddle_ocr = None


def get_paddle_ocr():
    """Create the local OCR engine once, after the web process has started."""
    global _paddle_ocr

    if _paddle_ocr is not None:
        return _paddle_ocr

    from paddleocr import PaddleOCR

    ocr_kwargs = {
        "device": "cpu",
        "text_detection_model_name": "PP-OCRv6_small_det",
        "text_recognition_model_name": "PP-OCRv6_small_rec",
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
    }

    if _DET_MODEL_DIR.exists():
        ocr_kwargs["text_detection_model_dir"] = str(_DET_MODEL_DIR)
    if _REC_MODEL_DIR.exists():
        ocr_kwargs["text_recognition_model_dir"] = str(_REC_MODEL_DIR)

    _paddle_ocr = PaddleOCR(**ocr_kwargs)
    return _paddle_ocr


# --------------------------------------------------
# 3. Helper functions
# --------------------------------------------------

def allowed_file(filename):
    """Check whether the uploaded file has an allowed extension."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def create_cleaned_image(image_path, output_path):
    """
    Create a readable, contrast-enhanced image.

    The original image remains unchanged. CLAHE is used
    only to produce the downloadable cleaned image.
    """
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError("The uploaded image could not be read.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    enhanced = clahe.apply(gray)

    if not cv2.imwrite(str(output_path), enhanced):
        raise ValueError("Could not save the cleaned image.")

    return output_path


def extract_text(image_path):
    """
    Extract text from the ORIGINAL uploaded image using PaddleOCR.

    PaddleOCR 3.x returns prediction results containing
    recognized text in the 'rec_texts' field.
    """
    results = get_paddle_ocr().predict(input=str(image_path))

    extracted_lines = []

    for result in results:
        result_data = result.json

        # PaddleOCR 3.x may expose json as a property.
        if callable(result_data):
            result_data = result_data()

        # Some versions wrap prediction data inside "res".
        if isinstance(result_data, dict):
            result_data = result_data.get("res", result_data)

        recognized_texts = result_data.get("rec_texts", [])

        extracted_lines.extend(
            text.strip()
            for text in recognized_texts
            if isinstance(text, str) and text.strip()
        )

    return "\n".join(extracted_lines)

def extract_text_openrouter(image_path):
    """Extract whiteboard text using Qwen2.5-VL via OpenRouter."""

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is missing. Check your .env file."
        )

    image_path = str(image_path)
    ext = image_path.rsplit(".", 1)[-1].lower() if "." in image_path else "jpeg"
    mime_map = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
    }
    mime_type = mime_map.get(ext) or mimetypes.guess_type(image_path)[0] or "image/jpeg"

    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(
            image_file.read()
        ).decode("utf-8")

    image_data_url = f"data:{mime_type};base64,{encoded_image}"

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    prompt = """
    You are EduScan, a whiteboard text extraction assistant.

    Transcribe all readable text in this image into structured Markdown.

    Requirements:
    - Preserve the original wording, spelling, numbers and equations.
    - Include headings, bullet points, diagrams' labels and tables where readable.
    - Keep the original reading order and logical line breaks.
    - Do not summarize, rewrite, or invent missing content.
    - Mark unreadable portions as [unclear].
    - Return only the extracted notes.
    """

    model_name = os.getenv("MODEL", "qwen/qwen-2.5-vl-72b-instruct").strip()

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url},
                    },
                ],
            }
        ],
        temperature=0,
        max_tokens=4096,
    )

    text = response.choices[0].message.content

    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Qwen returned an empty OCR result.")

    return text.strip()


def get_extracted_text(image_path):
    """
    Extract text using Qwen2.5-VL via OpenRouter as requested.
    Falls back to cached local PaddleOCR if the API key is not present or an error occurs.
    """
    if os.getenv("OPENROUTER_API_KEY"):
        try:
            app.logger.info("Extracting text with OpenRouter Qwen 2.5 VL...")
            return extract_text_openrouter(image_path)
        except Exception as err:
            app.logger.warning("OpenRouter extraction failed (%s), falling back to local OCR.", err)
    
    return extract_text(image_path)

def create_pdf(text, output_path):
    """Create a PDF from the user's edited transcription."""
    document = SimpleDocTemplate(str(output_path), pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    for line in text.splitlines():
        if line.strip():
            safe_text = escape(line)
            story.append(Paragraph(safe_text, styles["Normal"]))
        else:
            story.append(Spacer(1, 8))

    document.build(story)


# --------------------------------------------------
# 4. Main page: upload, process and edit
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    """Display the upload form and process uploaded images."""

    if request.method == "GET":
        return render_template(
            "index.html",
            original_image=None,
            cleaned_image=None,
            corrected_image=None,
            extracted_text="",
            file_id=None,
            perspective_applied=False,
            corners_found=False,
        )

    uploaded_file = request.files.get("image")

    if not uploaded_file or uploaded_file.filename == "":
        flash("Please select a whiteboard image first.", "error")
        return redirect(url_for("index"))

    if not allowed_file(uploaded_file.filename):
        flash("Please upload a PNG, JPG, JPEG or WebP image.", "error")
        return redirect(url_for("index"))

    # Create a unique filename to avoid accidental overwrites.
    extension = uploaded_file.filename.rsplit(".", 1)[1].lower()
    file_id = uuid4().hex
    safe_name = secure_filename(f"{file_id}.{extension}")

    original_path = UPLOAD_DIR / safe_name
    cleaned_name = f"{file_id}_cleaned.png"
    cleaned_path = OUTPUT_DIR / cleaned_name

    try:
        uploaded_file.save(str(original_path))

        # Validate that OpenCV can actually read the image.
        image = cv2.imread(str(original_path))
        if image is None:
            raise ValueError("This file is not a valid readable image.")

        # ----------------------------------------------------------------
        # Step A: Attempt automatic perspective correction.
        # detect_board_corners() returns ordered corners if a clear
        # quadrilateral whiteboard is found, otherwise None.
        # ----------------------------------------------------------------
        corners = detect_board_corners(image)
        perspective_applied = False
        corrected_image_url = None

        if corners is not None:
            try:
                corrected = correct_perspective(image, corners)

                if corrected is not None and corrected.size > 0:
                    corrected_name = f"{file_id}_corrected.jpg"
                    corrected_path = OUTPUT_DIR / corrected_name

                    if cv2.imwrite(str(corrected_path), corrected):
                        # Build the enhanced (cleaned) image from the
                        # perspective-corrected version.
                        create_cleaned_image(corrected_path, cleaned_path)
                        corrected_image_url = url_for(
                            "output_file", filename=corrected_name
                        )
                        perspective_applied = True
                        app.logger.info(
                            "Auto perspective correction applied for %s", file_id
                        )
            except Exception:
                app.logger.warning(
                    "Auto perspective correction failed for %s — "
                    "falling back to raw image.",
                    file_id,
                    exc_info=True,
                )

        # Step B: If auto-correction was not possible, build cleaned image
        # directly from the original.
        if not perspective_applied:
            create_cleaned_image(original_path, cleaned_path)

        # Step C: Run OCR on the best available image.
        ocr_source = (
            OUTPUT_DIR / f"{file_id}_corrected.jpg"
            if perspective_applied
            else original_path
        )
        text = get_extracted_text(ocr_source)

        if not text:
            text = "No text detected. Try a clearer image."

        return render_template(
            "index.html",
            original_image=url_for("uploaded_file", filename=safe_name),
            cleaned_image=url_for("output_file", filename=cleaned_name),
            corrected_image=corrected_image_url,
            extracted_text=text,
            file_id=file_id,
            perspective_applied=perspective_applied,
            corners_found=corners is not None,
        )

    except Exception as error:
        app.logger.exception("Image processing failed")
        flash(f"Could not process this image: {error}", "error")
        return redirect(url_for("index"))


# --------------------------------------------------
# 5. Serve uploaded and generated images
# --------------------------------------------------


# --------------------------------------------------
# 5A. Apply manual perspective correction
# --------------------------------------------------

@app.route("/correct-perspective", methods=["POST"])
def apply_perspective():
    """Correct a whiteboard using four manually selected corners."""

    file_id = request.form.get("file_id", "").strip()
    corners_json = request.form.get("corners", "")

    # Validate the file ID before using it to locate an uploaded image.
    if not file_id or len(file_id) != 32 or not all(
        character in "0123456789abcdef" for character in file_id.lower()
    ):
        flash("Invalid image reference. Please upload the image again.", "error")
        return redirect(url_for("index"))

    try:
        # Find the original uploaded image associated with this file ID.
        matching_files = [
            path for path in UPLOAD_DIR.glob(f"{file_id}.*")
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        ]

        if not matching_files:
            flash("Uploaded image not found. Please upload it again.", "error")
            return redirect(url_for("index"))

        original_path = matching_files[0]
        original_name = original_path.name

        # Convert the corner coordinates submitted by JavaScript into
        # the numeric format expected by OpenCV.
        corners_data = json.loads(corners_json)

        if not isinstance(corners_data, list) or len(corners_data) != 4:
            raise ValueError("Please select exactly four corners.")

        corners = np.asarray(
            [[point["x"], point["y"]] for point in corners_data],
            dtype=np.float32,
        )

        if not np.isfinite(corners).all():
            raise ValueError("Corner coordinates are invalid.")

        image = cv2.imread(str(original_path))
        if image is None:
            raise ValueError("The uploaded image could not be read.")

        height, width = image.shape[:2]

        # Ensure all selected points lie within the original image.
        if (
            (corners[:, 0] < 0).any()
            or (corners[:, 0] >= width).any()
            or (corners[:, 1] < 0).any()
            or (corners[:, 1] >= height).any()
        ):
            raise ValueError("One or more corners are outside the image.")

        # Reject invalid or nearly zero-area selections.
        if abs(cv2.contourArea(corners)) < 100:
            raise ValueError(
                "The selected area is too small. Please select the board corners."
            )

        # The expected order is top-left, top-right,
        # bottom-right, bottom-left.
        if not cv2.isContourConvex(corners.reshape(-1, 1, 2).astype("int32")):
            raise ValueError(
                "The corners do not form a valid quadrilateral. "
                "Select them clockwise around the board."
            )

        # Apply the existing perspective correction function.
        corrected_image = correct_perspective(image, corners)

        if corrected_image is None or corrected_image.size == 0:
            raise ValueError("Perspective correction did not produce an image.")

        # Save the corrected image separately, preserving the original.
        corrected_name = f"{file_id}_corrected.jpg"
        corrected_path = OUTPUT_DIR / corrected_name

        if not cv2.imwrite(str(corrected_path), corrected_image):
            raise ValueError("Could not save the corrected image.")

        # Generate the cleaned image from the corrected image.
        cleaned_name = f"{file_id}_cleaned.png"
        cleaned_path = OUTPUT_DIR / cleaned_name
        create_cleaned_image(corrected_path, cleaned_path)

        # Run OCR on the perspective-corrected image using OpenRouter Qwen 2.5 VL.
        text = get_extracted_text(corrected_path)
        if not text:
            text = "No text detected. Try a clearer image."

        return render_template(
            "index.html",
            original_image=url_for("uploaded_file", filename=original_name),
            cleaned_image=url_for("output_file", filename=cleaned_name),
            corrected_image=url_for("output_file", filename=corrected_name),
            extracted_text=text,
            file_id=file_id,
            perspective_applied=True,
            corners_found=True,
        )

    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        flash(f"Could not correct perspective: {error}", "error")
        return redirect(url_for("index"))

    except Exception:
        app.logger.exception("Perspective correction failed")
        flash(
            "Perspective correction failed. Check the terminal for details.",
            "error",
        )
        return redirect(url_for("index"))

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    """Return an uploaded image for preview."""
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/outputs/<filename>")
def output_file(filename):
    """Return a generated output image for preview or download."""
    return send_from_directory(OUTPUT_DIR, filename)


# --------------------------------------------------
# 6. Export edited text as TXT
# --------------------------------------------------

@app.route("/download/txt", methods=["POST"])
def download_txt():
    """Save the edited transcription as a text file."""
    text = request.form.get("extracted_text", "")

    if not text.strip():
        flash("There is no text to download.", "error")
        return redirect(url_for("index"))

    filename = f"eduscan_text_{uuid4().hex}.txt"
    output_path = OUTPUT_DIR / filename

    output_path.write_text(text, encoding="utf-8")

    return send_from_directory(
        OUTPUT_DIR,
        filename,
        as_attachment=True,
        download_name="eduscan_extracted_text.txt",
    )


# --------------------------------------------------
# 7. Export edited text as PDF
# --------------------------------------------------

@app.route("/download/pdf", methods=["POST"])
def download_pdf():
    """Save the edited transcription as a PDF."""
    text = request.form.get("extracted_text", "")

    if not text.strip():
        flash("There is no text to export.", "error")
        return redirect(url_for("index"))

    filename = f"eduscan_text_{uuid4().hex}.pdf"
    output_path = OUTPUT_DIR / filename

    try:
        create_pdf(text, output_path)

        return send_from_directory(
            OUTPUT_DIR,
            filename,
            as_attachment=True,
            download_name="eduscan_extracted_text.pdf",
        )

    except Exception:
        app.logger.exception("PDF generation failed")
        flash("Could not generate the PDF.", "error")
        return redirect(url_for("index"))


# --------------------------------------------------
# 8. Run the development server
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)