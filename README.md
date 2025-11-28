## AI Crime Evidence Organizer

**Full-featured AI-powered system** for organizing, analyzing, and extracting insights from criminal case evidence using state-of-the-art ML models.

### Features

✅ **Real AI Models Integrated:**
- **CLIP** for image classification (crime scene, CCTV, injury, weapon, environment)
- **BERT** for document classification (witness statements, medical reports, FIR, etc.)
- **YOLO** for object detection (persons, vehicles, weapons)
- **spaCy/BERT NER** for entity extraction (names, locations, dates)
- **OpenAI GPT** for intelligent reasoning and contradiction detection
- **Tesseract OCR** for text extraction from images and scanned PDFs

✅ **Complete Pipeline:**
- File type identification (magic bytes, mime types)
- Evidence classification (AI-powered)
- Content extraction (EXIF, OCR, objects, entities)
- Data normalization (timestamps, locations, names)
- Timeline construction
- Inconsistency detection (rule-based + LLM)
- Missing evidence recommendations
- Report generation (JSON, PDF)

### Project Structure
- `backend/` - FastAPI service with full AI model integration
- `frontend/` - HTML dashboard for uploads and reports
- `docs/` - Architecture and implementation guides
- `AI_Crime_Evidence_Organizer_Methodology.md` - Seminar-ready methodology

### Quick Start

#### 1. Install Dependencies

**Python packages:**
```bash
cd backend
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**System dependencies:**
- **Tesseract OCR**: [Windows](https://github.com/UB-Mannheim/tesseract/wiki) | Linux: `sudo apt-get install tesseract-ocr` | macOS: `brew install tesseract`
- **Poppler** (for PDF OCR): [Windows](https://github.com/oschwartz10612/poppler-windows/releases) | Linux: `sudo apt-get install poppler-utils` | macOS: `brew install poppler`

#### 2. Configure Environment

```bash
cd backend
cp env.example .env
```

Edit `.env` and set:
```
OPENAI_API_KEY=sk-your-key-here
EVIDENCE_ENABLE_REAL_AI=true
```
Add optional toggles to disable object detection and image classification (document-first processing):
```
EVIDENCE_ENABLE_OBJECT_DETECTION=true
EVIDENCE_ENABLE_IMAGE_CLASSIFICATION=true
```

#### 3. Run Server

```bash
uvicorn app.main:app --reload
```

#### 4. Use the System

Open `frontend/index.html` in your browser and upload evidence files.

### Models

Models download automatically on first use (~600MB total):
- CLIP: ~150MB (image classification)
- BERT: ~440MB (document classification & NER)
- YOLO: ~6MB (object detection)

### Documentation

- **Setup Guide**: `backend/SETUP.md`
- **Full Implementation Guide**: `docs/IMPLEMENTATION_GUIDE.md`
- **Architecture**: `docs/architecture.md`
- **Methodology**: `AI_Crime_Evidence_Organizer_Methodology.md`

### API Endpoints

- `POST /evidence/upload` - Upload evidence file
- `POST /evidence/process` - Process evidence through AI pipeline
 - `POST /evidence/upload_folder` - Upload a folder (multiple files or a zip) into a new case
 - `POST /evidence/process_case` - Process a full case (all files in `storage/case_data/{case_id}/raw_files/`)
- `GET /reports/{case_id}` - Get analysis report (JSON)
- `GET /reports/{case_id}/pdf` - Download PDF report

### System Requirements

- Python 3.9+
- 8GB+ RAM (16GB recommended)
- CUDA GPU (optional, for faster inference)
- 10GB+ disk space

### Next Steps

- Fine-tune models on your crime case dataset
- Add face recognition and license plate detection
- Integrate with police databases
- Deploy to production with Docker/Kubernetes

For detailed implementation and customization, see `docs/IMPLEMENTATION_GUIDE.md`.

