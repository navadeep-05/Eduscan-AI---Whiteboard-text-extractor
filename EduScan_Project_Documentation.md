# EduScan — Classroom Whiteboard Text Extractor

**Project documentation**

---

## Project Description

EduScan is a Flask web application that turns a classroom whiteboard photograph into editable notes. A user uploads a PNG, JPG, JPEG or WebP image (up to 15 MB). The server attempts to find the board as a four-corner quadrilateral, warps it to a flat rectangle when detection succeeds, builds a contrast-enhanced download image with CLAHE, transcribes the writing, and lets the user edit the text before exporting TXT or PDF.

The product name in the UI is **EduScan AI**. The live pipeline is implemented in `app.py` (HTTP routes, OCR, PDF) and `image_processing.py` (board detection and perspective warp). The page is `templates/index.html`, styled by `static/style.css`. Manual corner picking is `static/perspective.js`.

Two OCR paths exist:

1. **Primary (when an API key is present):** a vision-language model via OpenRouter. Default model: `qwen/qwen-2.5-vl-72b-instruct` (`MODEL` in `.env`).
2. **Local fallback:** PaddleOCR 3.x on CPU, using cached **PP-OCRv6 Small** detection and recognition weights under `models/`.

The Hugging Face whiteboard set is **not** loaded by the Flask app. It was used to evaluate OCR engines (character and word error rates) during development.

### Scenarios

- A teacher photographs a tilted classroom board and wants a flattened, higher-contrast copy plus typed notes.
- Auto-detection fails (glare, weak border, clutter). The user clicks four corners and re-runs warp + OCR.
- OpenRouter is unavailable. The same image is transcribed with local PP-OCRv6 Small.

---

## Dataset used

| Item | Value |
|---|---|
| Name | **Whiteboards (OCR / VLM eval)** |
| Hugging Face id | **`danielrosehill/Whiteboards`** |
| Local folder (this copy) | `HF dataset/whiteboards_hf/` |
| License | CC-BY-4.0 |
| Size | **21** phone photos, WebP, about 4096×3072, ~150 KB each |
| Split | All rows are `test` (`data/metadata.jsonl`) |
| Ground truth fields | `file_name`, `id`, `category`, `transcription`, `description` |
| Categories | prose (2), list (6), table (2), flowchart (1), diagram (9), mixed (1) |
| Purpose in this project | Offline OCR comparison (EasyOCR vs PaddleOCR) using `transcription` as reference |
| Used at Flask runtime? | **No.** Uploads go to `static/uploads/` |

Source: [huggingface.co/datasets/danielrosehill/Whiteboards](https://huggingface.co/datasets/danielrosehill/Whiteboards) and the bundled `HF dataset/whiteboards_hf/README.md`.

**What was measured on this set (development, original workspace):** on all 21 images, EasyOCR CER/WER ≈ 69% / 99%; PaddleOCR Small CER/WER ≈ 61% / 84%. CLAHE vs original pixels changed PaddleOCR error only slightly. That is why the app treats CLAHE as a **downloadable enhanced image**, and why production transcription prefers a VLM with Paddle as fallback.

Lab photos under `assets/` (older copies of the project) were local OpenCV exercises, not the evaluation dataset.

---

## PaddleOCR models used

Configured in `app.py` and stored locally so the first run does not re-download:

| Role | Model name | Folder | Hugging Face |
|---|---|---|---|
| Text **detection** | **`PP-OCRv6_small_det`** | `models/PP-OCRv6_small_det/` | [PaddlePaddle/PP-OCRv6_small_det](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det) |
| Text **recognition** | **`PP-OCRv6_small_rec`** | `models/PP-OCRv6_small_rec/` | [PaddlePaddle/PP-OCRv6_small_rec](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec) |

Runtime flags: `device="cpu"`, document orientation / unwarping / textline orientation **off** (EduScan already warps the board with OpenCV).

PaddleOCR 3.x `predict()` results are read from `rec_texts` (with a possible `res` wrapper).

**Earlier experiment (not in the live app):** EasyOCR English CPU reader in `ocr_engine.py` on older project copies. Do not treat EasyOCR as the current extractor.

**Cloud OCR (primary when configured):** OpenRouter OpenAI-compatible API, default `qwen/qwen-2.5-vl-72b-instruct`.

---

## Technical architecture

```
Browser (index.html + style.css + perspective.js)
        |  POST multipart image
        v
Flask app.py
        |-- image_processing.detect_board_corners  (Canny + contours)
        |-- image_processing.correct_perspective   (getPerspectiveTransform)
        |-- CLAHE cleaned PNG                      (OpenCV)
        |-- get_extracted_text
        |      |-- OpenRouter Qwen 2.5 VL  (if OPENROUTER_API_KEY)
        |      +-- PaddleOCR PP-OCRv6 Small (fallback)
        v
Editor textarea  -->  TXT download  |  ReportLab PDF download
```

**Routes**

| Method | Path | Role |
|---|---|---|
| GET/POST | `/` | Upload form; process image; render results |
| POST | `/correct-perspective` | Manual four-corner warp, then OCR |
| GET | `/uploads/<filename>` | Original preview |
| GET | `/outputs/<filename>` | Cleaned / corrected / export files |
| POST | `/download/txt` | Edited text as `.txt` |
| POST | `/download/pdf` | Edited text as A4 PDF |

**Important files (read these to understand the system)**

| File | Why it matters |
|---|---|
| `app.py` | Entire backend: upload limits, Paddle init, Qwen prompt, CLAHE, exports |
| `image_processing.py` | `detect_board_corners`, `order_board_corners`, `correct_perspective` |
| `templates/index.html` | Upload, comparison grid, manual warp UI, editor, export buttons |
| `static/perspective.js` | Canvas clicks → original-image coordinates → JSON corners |
| `static/style.css` | Layout and theme |
| `requirements.txt` | Python packages (includes leftover EasyOCR/matplotlib/jiwer) |
| `HF dataset/whiteboards_hf/README.md` | Dataset definition |
| `HF dataset/whiteboards_hf/data/metadata.jsonl` | 21 ground-truth transcriptions |
| `models/PP-OCRv6_small_det/README.md` | Detection model card |
| `models/PP-OCRv6_small_rec/README.md` | Recognition model card |
| `.env` | `OPENROUTER_API_KEY`, `MODEL` — **local only, never zip this** |

---

## Prerequisites

- Python **3.11** recommended (the Paddle environment used during development).
- Packages from `requirements.txt` that the **live app** actually imports: `flask`, `opencv-python`, `numpy`, `paddleocr`, `reportlab`, plus `openai` and `python-dotenv` (used in `app.py` but not listed in `requirements.txt` yet).
- Optional: OpenRouter API key for Qwen. Without it, only local PaddleOCR runs.
- Cached folders `models/PP-OCRv6_small_det` and `models/PP-OCRv6_small_rec` (or allow PaddleOCR to download the same names).
- Run from the project root: `python app.py` (Flask debug server). Open the printed localhost URL.

---

## Project workflow

### Milestone 1: Image intake and board geometry

**Activity 1.1 — Accept a whiteboard photo**  
Validate extension and size. Save under `static/uploads/{uuid}.{ext}`. Reject unreadable files via `cv2.imread`.

**Activity 1.2 — Detect four board corners**  
Grayscale, blur, Canny, morphological close, largest convex 4-gons covering at least 10% of the image, aspect ratio ≤ 5. Order corners TL, TR, BR, BL.

**Activity 1.3 — Warp or skip**  
If corners exist, `warpPerspective` to a rectangle and save `{uuid}_corrected.jpg`. Otherwise OCR uses the original upload.

### Milestone 2: Visual enhancement

**Activity 2.1 — CLAHE cleaned image**  
Always write `{uuid}_cleaned.png` (clipLimit 2.0, 8×8 tiles) from the corrected image when warp succeeded, else from the original. This is the downloadable “enhanced” view, not the only OCR input.

### Milestone 3: Text extraction

**Activity 3.1 — Vision-language transcription**  
Encode the OCR source image as a data URL. Prompt EduScan to copy wording, equations, lists and tables into Markdown, mark unreadable spots as `[unclear]`, and not invent content.

**Activity 3.2 — Local PaddleOCR Small fallback**  
`ocr.predict()` with PP-OCRv6 small det/rec; join non-empty `rec_texts`.

### Milestone 4: Evaluation dataset (development)

**Activity 4.1 — Load `danielrosehill/Whiteboards`**  
21 WebP frames + `transcription` in `metadata.jsonl`.

**Activity 4.2 — Compare engines**  
CER/WER on normalized text. PaddleOCR Small outperformed EasyOCR on this set; CLAHE did not fix handwriting error enough to rely on classical OCR alone. That result justifies Qwen-first + Paddle fallback.

### Milestone 5: Web interface

**Activity 5.1 — Upload workspace**  
Hero, drag-and-drop zone, 15 MB hint, processing overlay (save → detect → warp → CLAHE → OCR).

**Activity 5.2 — Results**  
Original vs enhanced, auto-warp banner, optional manual corners, editable textarea, copy, TXT/PDF.

### Milestone 6: Export and local run

**Activity 6.1 — TXT/PDF**  
Exports use the **current editor text**, not the raw OCR string. PDF uses ReportLab on A4 with XML-escaped paragraphs.

**Activity 6.2 — Run locally**  
Project root, virtualenv with PaddleOCR, `.env` for OpenRouter, `python app.py`.

---

## Exploring the application’s web pages

There is a **single page** (`/`) that changes after upload.

1. **Landing / upload** — branding, feature list, file picker (PNG/JPG/JPEG/WEBP).
2. **Processing overlay** — five labelled steps matching the backend.
3. **Results comparison** — original vs CLAHE (and warp if applied).
4. **Manual perspective** — optional canvas; corners posted to `/correct-perspective`.
5. **Extracted text** — editable notes, copy, Download TXT, Download PDF.
6. **Footer** — stack noted as Python, Flask, OpenCV and Qwen 2.5-VL.

Flash messages cover empty upload, bad type, failed warp, empty export, and processing errors.

---

## Files to keep vs drop (for a simple, documentable folder)

**Keep (needed to understand and run the product)**

- `app.py`
- `image_processing.py`
- `templates/index.html`
- `static/style.css`
- `static/perspective.js`
- `static/uploads/` and `static/outputs/` as **empty directories**
- `requirements.txt` (then trim unused lines)
- `models/PP-OCRv6_small_det/` and `models/PP-OCRv6_small_rec/` **to run** local OCR (weights are large; omit from a “docs zip” but keep on disk)
- `HF dataset/whiteboards_hf/README.md` and `data/metadata.jsonl` **to document the evaluation set**
- This documentation file
- `.gitignore` (should ignore `.env`, venvs, `__pycache__`, uploads/outputs contents)

**Keep one or two sample images only if you want screenshots in a report** — e.g. `HF dataset/whiteboards_hf/data/01.webp`. You do not need all 21 files to explain the dataset.

**Do not keep in a zip you send to another AI or share publicly**

- `.env` (API key)
- Paddle **weight binaries** (`inference.pdiparams` and similar) if the zip is only for writing docs
- Entire `HF dataset/.../data/*.webp` set (unless you still need to rerun CER/WER)
- `venv` / `venv_paddle` from any older copy
- `ocr_engine.py`, `test_paddleocr.py`, benchmark CSVs, `assets/` lab dumps, `extracted_text.pdf` (already removed from this copy)

**Nested folder note:** this tree is `CV Classboard textractor / CV Classboard text extractor / ...`. For sharing, zip the **inner** project so files sit at the zip root.

---

## Conclusion

EduScan is a whiteboard capture tool: OpenCV finds and flattens the board, CLAHE produces a readable still, and text is taken first from **Qwen 2.5-VL (OpenRouter)** with **PP-OCRv6 Small** (`PP-OCRv6_small_det` + `PP-OCRv6_small_rec`) on CPU as backup. The evaluation corpus is Hugging Face **`danielrosehill/Whiteboards`** (21 labelled boards). The Flask app itself only needs the Python/HTML/CSS/JS sources, optional local models, and a local `.env` for the cloud model.

Always review OCR output before treating it as the official class notes.
