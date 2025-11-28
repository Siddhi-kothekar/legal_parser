# AI Crime Evidence Organizer - Full Implementation Guide

## Overview

This guide explains how to implement and use the real AI models (CLIP, BERT, YOLO, LLM) in the AI Crime Evidence Organizer system.

## Prerequisites

### System Requirements
- Python 3.9 or higher
- CUDA-capable GPU (recommended for faster inference, but CPU works)
- At least 8GB RAM (16GB+ recommended)
- 10GB+ free disk space for models

### Software Dependencies

#### 1. Python Packages
Install all Python dependencies:
```bash
cd backend
pip install -r requirements.txt
```

#### 2. System Dependencies

**For OCR (Tesseract):**
- **Windows**: Download installer from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
- **Linux**: `sudo apt-get install tesseract-ocr`
- **macOS**: `brew install tesseract`

**For PDF OCR (poppler):**
- **Windows**: Download from [poppler-windows releases](https://github.com/oschwartz10612/poppler-windows/releases)
- **Linux**: `sudo apt-get install poppler-utils`
- **macOS**: `brew install poppler`

**For python-magic (file type detection):**
- **Windows**: Install [python-magic-bin](https://pypi.org/project/python-magic-bin/)
- **Linux**: `sudo apt-get install libmagic1`
- **macOS**: `brew install libmagic`

#### 3. spaCy Language Model
Download the English language model:
```bash
python -m spacy download en_core_web_sm
# For better accuracy (optional):
python -m spacy download en_core_web_lg
```

## Configuration

### 1. Environment Variables

Create a `.env` file in the `backend` directory:

```bash
cp backend/.env.example backend/.env
```

Edit `backend/.env` and set:
- `OPENAI_API_KEY`: Your OpenAI API key (required for LLM reasoning)
- `EVIDENCE_ENABLE_REAL_AI=true`: Enable real AI models
- `EVIDENCE_DEVICE=auto`: Use GPU if available, else CPU

### 2. Model Download

Models are downloaded automatically on first use:
- **CLIP**: ~150MB (from Hugging Face)
- **BERT**: ~440MB (from Hugging Face)
- **YOLO**: ~6MB (from Ultralytics)

Total download: ~600MB on first run.

## Model Implementation Details

### 1. CLIP for Image Classification

**Location**: `backend/app/services/model_manager.py`

**Usage**:
```python
from app.services.model_manager import model_manager

result = model_manager.classify_image_clip(image_path)
# Returns: {"label": "cctv", "confidence": 0.92, "all_scores": {...}}
```

**How it works**:
1. Loads CLIP model (`openai/clip-vit-base-patch32`)
2. Processes image and text prompts together
3. Computes similarity scores between image and category prompts
4. Returns best matching category with confidence

**Categories**:
- `crime_scene`: Crime scene photographs
- `cctv`: CCTV surveillance frames
- `injury`: Medical/injury images
- `weapon`: Weapon photographs
- `environment`: Environmental evidence

**Customization**:
- Change model: Edit `CLIP_MODEL_NAME` in `model_manager.py`
- Add categories: Add prompts to `IMAGE_CLASS_PROMPTS`

### 2. BERT for Document Classification

**Location**: `backend/app/services/model_manager.py`

**Usage**:
```python
result = model_manager.classify_document_bert(text)
# Returns: {"label": "witness_statement", "confidence": 0.87}
```

**How it works**:
1. Extracts BERT embeddings for document text
2. Computes embeddings for category prompts
3. Uses cosine similarity to find best match
4. Returns category with confidence score

**Categories**:
- `witness_statement`: Witness testimonies
- `medical_report`: Medical examination reports
- `fir`: First Information Report
- `police_memo`: Internal police memos
- `legal_document`: Other legal documents

**Fine-tuning** (Advanced):
To improve accuracy, fine-tune BERT on your legal document dataset:
```python
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=5  # 5 document categories
)
# Train on your labeled dataset...
```

### 3. YOLO for Object Detection

**Location**: `backend/app/services/model_manager.py`

**Usage**:
```python
objects = model_manager.detect_objects_yolo(image_path)
# Returns: [{"class": "person", "confidence": 0.95, "bbox": [x1, y1, x2, y2]}, ...]
```

**How it works**:
1. Loads YOLOv8n (nano) model
2. Runs inference on image
3. Filters for crime-relevant objects
4. Returns bounding boxes with confidence scores

**Detected Objects**:
- Persons, vehicles (car, truck, motorcycle, bus)
- Weapons (knife, gun)
- Items (bottle, backpack, handbag)

**Note:** Detecting blood stains, injured body areas, or subtle forensic cues requires a fine-tuned or custom YOLO dataset trained on labeled crime scene imagery. For high accuracy, create a small annotated dataset and retrain YOLO with the specific classes (e.g., `blood`, `injured_limb`, `stab_wound`) and increase the model size from `yolov8n` to `yolov8m` or larger.

**Customization**:
- Use larger model: Change `"yolov8n.pt"` to `"yolov8m.pt"` or `"yolov8l.pt"`
- Train custom model: Fine-tune on crime-specific objects
- Adjust confidence threshold: Edit `conf > 0.3` in `detect_objects_yolo()`

### 4. spaCy/BERT for Named Entity Recognition

**Location**: `backend/app/services/model_manager.py`

**Usage**:
```python
entities = model_manager.extract_entities_bert(text)
# Returns: [{"entity": "John Doe", "label": "PERSON", ...}, ...]
```

**How it works**:
1. Uses spaCy's `en_core_web_sm` model (fast, reliable)
2. Extracts standard NER labels: PERSON, ORG, GPE, DATE, TIME
3. Returns entities with positions

**Custom Entities** (Advanced):
Train a custom NER model for legal entities:
```python
# Use spaCy's training pipeline
# Or use transformers with a fine-tuned BERT-NER model
```

### 5. OpenAI LLM for Reasoning

**Location**: `backend/app/services/model_manager.py` and `reasoning.py`

**Usage**:
```python
response = model_manager.llm_reasoning(
    prompt="Analyze this evidence...",
    system_prompt="You are a crime analyst...",
    model="gpt-4o-mini"
)
```

### Folder Upload & Case Processing

The server supports uploading a complete folder of mixed evidence as either a ZIP archive or multiple files in a single multipart request. The folder is stored under `storage/case_data/{case_id}/raw_files/` and processed with a single case-level pipeline run.

API endpoints:
- `POST /evidence/upload_folder` — Accepts multiple files or a zip archive. Returns a `case_id` and stored paths.
- `POST /evidence/process_case` — Triggers the AI pipeline for an entire case (case-level aggregated timeline and report). Accepts `case_id` and optional `artifact_paths` to process a subset.
- `GET /evidence/cases/{case_id}/files` — Returns a list of files stored for a case.

Notes:
- Upload zone: `storage/case_data/{case_id}/raw_files/`.
- The processing flow aggregates data across all artifacts and produces one consolidated report for the case.
- For single-file uploads, use `POST /evidence/upload` followed by `POST /evidence/process` (backwards-compatible).

**How it works**:
1. Sends prompt to OpenAI API
2. Uses GPT-4o-mini for cost-effective reasoning
3. Returns structured JSON responses

**Applications**:
- **Inconsistency Detection**: Complex contradiction analysis
- **Missing Evidence Suggestions**: Intelligent recommendations

**Cost Optimization**:
- Use `gpt-4o-mini` for most tasks (cheaper)
- Use `gpt-4` or `gpt-4-turbo` for critical analysis (better quality)
- Cache responses for similar queries

## Service Integration

### Classification Service

**File**: `backend/app/services/classification.py`

**Flow**:
1. Checks if real AI is enabled
2. Routes images to CLIP, documents to BERT
3. Falls back to heuristics if models fail

**Example**:
```python
from app.services.classification import ClassificationService

classifier = ClassificationService()
result = classifier.classify(image_path, extracted_text="")
```

### Extraction Service

**File**: `backend/app/services/extraction.py`

**Flow**:
1. **Images**: EXIF → YOLO → OCR
2. **Documents**: PDF text → BERT NER → Event extraction

**Example**:
```python
from app.services.extraction import ExtractionService

extractor = ExtractionService()
data = extractor.extract(document_path)
```

### Reasoning Service

**File**: `backend/app/services/reasoning.py`

**Flow**:
1. Rule-based checks (time, location, injury conflicts)
2. LLM-based complex reasoning
3. Missing evidence suggestions

**Example**:
```python
from app.services.reasoning import ReasoningService

reasoner = ReasoningService()
inconsistencies = reasoner.detect_inconsistencies(events, extracted_data)
suggestions = reasoner.suggest_missing_evidence(events, extracted_data)
```

## Performance Optimization

### 1. GPU Acceleration

**Enable CUDA**:
```python
# In model_manager.py, device is auto-detected
# Ensure CUDA is available:
import torch
print(torch.cuda.is_available())  # Should be True
```

**Memory Management**:
- Models are lazy-loaded (only when needed)
- Use smaller models for CPU: `yolov8n`, `clip-vit-base-patch32`
- Batch processing for multiple files

### 2. Caching

**Model Caching**:
Models are cached in memory after first load. Restart server to clear.

**Response Caching** (Future):
Implement Redis cache for LLM responses to reduce API costs.

### 3. Batch Processing

**Process Multiple Files**:
```python
# In pipeline.py, extend to handle multiple files
for file_path in evidence_files:
    pipeline.execute(case_id, file_path)
```

## Testing

### 1. Test Individual Models

```python
# Test CLIP
from app.services.model_manager import model_manager
result = model_manager.classify_image_clip(Path("test_image.jpg"))
print(result)

# Test BERT
result = model_manager.classify_document_bert("This is a witness statement...")
print(result)

# Test YOLO
objects = model_manager.detect_objects_yolo(Path("test_image.jpg"))
print(objects)

# Test LLM
response = model_manager.llm_reasoning("What is 2+2?")
print(response)
```

### 2. Test Full Pipeline

```python
from app.services.pipeline import EvidencePipeline

pipeline = EvidencePipeline()
pipeline.execute("case_123", Path("evidence.pdf"))
```

### 3. API Testing

```bash
# Start server
cd backend
uvicorn app.main:app --reload

# Upload evidence
curl -X POST "http://localhost:8000/evidence/upload" \
  -F "file=@test_evidence.pdf"

# Process evidence
curl -X POST "http://localhost:8000/evidence/process" \
  -H "Content-Type: application/json" \
  -d '{"case_id": "...", "artifact_path": "..."}'

# Get report
curl "http://localhost:8000/reports/{case_id}"
```

## Troubleshooting

### Model Loading Errors

**Issue**: `CUDA out of memory`
**Solution**: 
- Use CPU: Set `EVIDENCE_DEVICE=cpu`
- Use smaller models
- Process files one at a time

**Issue**: `Model not found`
**Solution**:
- Check internet connection (models download from Hugging Face)
- Clear cache: `rm -rf ~/.cache/huggingface/`

### OCR Errors

**Issue**: `TesseractNotFoundError`
**Solution**: Install Tesseract (see Prerequisites)

**Issue**: Poor OCR accuracy
**Solution**:
- Use higher resolution images
- Preprocess images (contrast, denoising)
- Use Google Vision API for better accuracy

### LLM API Errors

**Issue**: `OpenAI API key not set`
**Solution**: Set `OPENAI_API_KEY` in `.env`

**Issue**: `Rate limit exceeded`
**Solution**:
- Add retry logic
- Use cheaper model (`gpt-4o-mini`)
- Implement request queuing

## Advanced Customization

### 1. Fine-tune Models

**CLIP Fine-tuning**:
```python
# Use Hugging Face transformers training pipeline
from transformers import CLIPModel, CLIPProcessor, Trainer

# Load your labeled dataset
# Train on crime-specific image categories
```

**BERT Fine-tuning**:
```python
# Fine-tune for legal document classification
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=5
)
# Train on your document dataset
```

### 2. Custom Object Detection

**Train YOLO on Custom Objects**:
```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data="custom_dataset.yaml", epochs=100)
```

### Legal entity extraction

The pipeline extracts legal entities from document text using a combination of spaCy/BERT NER and regex heuristics. Entities include:
- IPC sections (e.g., IPC 302)
- FIR numbers (e.g., FIR 123/2025)
- MLC numbers
- Hospital, clinic, or medical center names
- Vehicle registration numbers
- Injury descriptions (e.g., 'stab wound', 'deep laceration', 'laceration 7 cm')

These entities are added to the `legal_entities` field in the JSON report. You can view them in the frontend under 'Extracted Content'.


### Image handling configuration (document-first pipeline)

If the project must rely primarily on documents and OCR (not object detection), use the following environment toggles to disable YOLO & CLIP and let the pipeline prioritize documents:

Environment settings (backend/.env):
```bash
# Use `true` or `false` to enable or disable features
EVIDENCE_ENABLE_REAL_AI=true  # keep BERT/LLM models active
EVIDENCE_ENABLE_OBJECT_DETECTION=false  # skip YOLO detections
EVIDENCE_ENABLE_IMAGE_CLASSIFICATION=false  # skip CLIP image classification
```

When these toggles are set to `false`, images will still be processed for EXIF and OCR text (timestamps and labels), but will **not** run YOLO or CLIP. This ensures the pipeline is document-first and avoids noisy or unreliable object detections when images are low quality.


### 3. Custom NER Model

**Train spaCy NER**:
```python
import spacy
from spacy.training import Example

nlp = spacy.blank("en")
# Add NER component
# Train on legal entity dataset
```

## Production Deployment

### 1. Model Serving

**Option A: Local Models**
- Deploy on GPU server
- Use model caching
- Implement request queuing

**Option B: Model API**
- Use Hugging Face Inference API
- Use OpenAI API for LLM
- Use cloud vision APIs (Google Vision, AWS Rekognition)

### 2. Scalability

**Horizontal Scaling**:
- Use Celery for background processing
- Deploy multiple workers
- Use Redis for task queue

**Vertical Scaling**:
- Use larger GPU instances
- Batch process multiple files
- Cache model outputs

### 3. Monitoring

**Metrics to Track**:
- Model inference time
- API response times
- Error rates
- GPU/CPU utilization
- API costs (OpenAI)

## Cost Estimation

### Model Inference (Local)
- **Free**: Running models on your own hardware
- **GPU Cost**: ~$0.50-2.00/hour for cloud GPU

### OpenAI API
- **gpt-4o-mini**: ~$0.15 per 1M input tokens, $0.60 per 1M output tokens
- **gpt-4**: ~$30 per 1M input tokens, $60 per 1M output tokens
- **Typical case**: ~$0.10-0.50 per case analysis

### Storage
- Model files: ~600MB
- Evidence files: Variable
- Reports: ~1-10MB per case

## Next Steps

1. **Fine-tune Models**: Train on your specific crime case data
2. **Add More Models**: Integrate face recognition, license plate detection
3. **Improve Accuracy**: Use larger models, ensemble methods
4. **Optimize Performance**: Batch processing, caching, GPU optimization
5. **Add Features**: Real-time processing, mobile app integration

## Support

For issues or questions:
1. Check this guide
2. Review code comments in service files
3. Check model documentation (Hugging Face, Ultralytics, OpenAI)
4. Open an issue on the project repository

