# 📸 EduScan — AI-Powered Classroom Board Text Extractor

EduScan is a web application that extracts handwritten or printed text from classroom whiteboard/blackboard photos. Upload a photo, get a clean perspective-corrected image, and receive an editable transcription — all from your browser.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web_App-black?logo=flask)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-green?logo=opencv)
![PaddleOCR](https://img.shields.io/badge/PaddleOCR-Text_Recognition-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

| Feature | Description |
|---|---|
| 📤 **Image Upload** | Upload whiteboard/blackboard photos (PNG, JPG, JPEG, WebP, up to 15 MB) |
| 🔲 **Auto Perspective Correction** | Automatically detects board edges and flattens tilted images |
| ✋ **Manual Corner Selection** | Click four corners on the image to manually correct perspective when auto-detection fails |
| 🖼️ **Image Enhancement** | CLAHE-based contrast enhancement produces a clean, readable grayscale version |
| 🔍 **Dual OCR Engine** | Primary: Qwen 2.5 VL 72B (via OpenRouter API) · Fallback: PaddleOCR v6 (local, offline) |
| ✏️ **Editable Transcription** | Edit the extracted text directly in the browser before exporting |
| 📄 **Export to TXT / PDF** | Download the final transcription as a `.txt` or `.pdf` file |

---

## 🛠️ Tech Stack

- **Backend:** Python 3.10+, Flask
- **Computer Vision:** OpenCV (edge detection, perspective transform, CLAHE enhancement)
- **OCR (Local):** PaddleOCR v6 Small (detection + recognition models, runs on CPU)
- **OCR (Cloud):** Qwen 2.5 VL 72B via [OpenRouter](https://openrouter.ai/) API
- **PDF Generation:** ReportLab
- **Frontend:** HTML/CSS + vanilla JavaScript (perspective corner picker)

---

## 📁 Project Structure

```
EduScan/
├── app.py                 # Flask application (routes, OCR, export)
├── image_processing.py    # Board detection, perspective correction, enhancement
├── requirements.txt       # Python dependencies
├── .env                   # API keys (not tracked by git)
├── .gitignore
├── models/                # PaddleOCR model cache (auto-downloaded on first run)
│   ├── PP-OCRv6_small_det/
│   └── PP-OCRv6_small_rec/
├── static/
│   ├── style.css          # Stylesheet
│   └── perspective.js     # Manual corner selection UI
├── templates/
│   └── index.html         # Main web UI
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11** (required on Render; PaddlePaddle has no Python 3.14 wheels)
- **pip** (Python package manager)
- *(Optional)* An [OpenRouter](https://openrouter.ai/) API key for cloud-based OCR

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/eduscan.git
cd eduscan
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
# Required for cloud OCR (Qwen 2.5 VL via OpenRouter)
OPENROUTER_API_KEY=sk-or-v1-your-api-key-here

# Optional: change the model (defaults to qwen/qwen-2.5-vl-72b-instruct)
MODEL=qwen/qwen-2.5-vl-72b-instruct
```

> [!NOTE]
> If you skip the `.env` file, EduScan will still work using the **local PaddleOCR** engine (no API key needed). Cloud OCR generally produces better results for handwritten text.

### 5. Run the App

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

---

## 📖 How It Works

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Upload     │────▶│  Board Detection │────▶│  Perspective    │
│  Image      │     │  (Canny + Contour)│    │   Correction     │
└─────────────┘     └──────────────────┘     └────────┬─────────┘
                                                       │
                    ┌──────────────────┐     ┌─────────▼─────────┐
                    │  Export          │◀───│  OCR Extraction   │
                    │  (TXT / PDF)     │     │  (Qwen VL /       │
                    └──────────────────┘     │   PaddleOCR)      │
                                             └───────────────────┘
```

1. **Upload** — The user uploads a whiteboard photo through the web interface.
2. **Board Detection** — OpenCV attempts to automatically find the board edges using Canny edge detection and contour approximation.
3. **Perspective Correction** — If four corners are detected (or manually selected), the image is flattened using a perspective transform.
4. **Image Enhancement** — CLAHE (Contrast Limited Adaptive Histogram Equalization) produces a clean grayscale version for download.
5. **Text Extraction** — The corrected image is sent to Qwen 2.5 VL (cloud) or PaddleOCR (local) for text recognition.
6. **Edit & Export** — The transcription is displayed in an editable text area. The user can correct any errors and export as TXT or PDF.

---

## 🔧 Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | *(none)* | API key from [OpenRouter](https://openrouter.ai/). If absent, local PaddleOCR is used. |
| `MODEL` | `qwen/qwen-2.5-vl-72b-instruct` | Vision-language model for cloud OCR. |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m "Add my feature"`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) — Local OCR engine
- [OpenRouter](https://openrouter.ai/) — Cloud AI model gateway
- [Qwen 2.5 VL](https://huggingface.co/Qwen) — Vision-language model for handwriting recognition
- [OpenCV](https://opencv.org/) — Computer vision library
- [Flask](https://flask.palletsprojects.com/) — Python web framework
- [ReportLab](https://www.reportlab.com/) — PDF generation

