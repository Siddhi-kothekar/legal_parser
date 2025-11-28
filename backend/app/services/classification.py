"""
Step 3: Evidence Classification using real AI models.

Uses:
- CLIP for image classification
- BERT for document classification
"""

from pathlib import Path
from typing import Dict, Any

from app.config import settings
from app.services.model_manager import model_manager


class ClassificationService:
    """
    Evidence Classification Service using CLIP (images) and BERT (documents).
    """

    def classify(self, artifact_path: Path, extracted_text: str = "") -> Dict[str, Any]:
        """
        Classify evidence file using AI models.
        
        Args:
            artifact_path: Path to the evidence file
            extracted_text: Pre-extracted text for documents (optional, will extract if not provided)
        
        Returns:
            Dict with 'label', 'confidence', and optional 'all_scores'
        """
        extension = artifact_path.suffix.lower()

        if settings.enable_real_ai:
            if extension in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
                # Respect enable_image_classification toggle
                if getattr(settings, 'enable_image_classification', True):
                    return self._classify_image_ai(artifact_path)
                else:
                    # Fallback: use OCR text if available for classification
                    ocr_text = ""
                    try:
                        from app.services.extraction import ExtractionService
                        ocr_text = ExtractionService()._extract_image_metadata(artifact_path).get("ocr_text", "")
                    except Exception:
                        ocr_text = ""
                    # Reuse document classifier path if OCR provides text
                    if ocr_text and len(ocr_text.strip()) > 5:
                        return self._classify_document_ai(ocr_text, artifact_path)
                    # Heuristic fallback
                    return self._classify_heuristic(artifact_path)
            else:
                # For documents, we need text first
                if not extracted_text:
                    extracted_text = self._quick_text_extract(artifact_path)
                return self._classify_document_ai(extracted_text, artifact_path)
        else:
            # Fallback to heuristic-based classification
            return self._classify_heuristic(artifact_path)

    def _classify_image_ai(self, artifact_path: Path) -> Dict[str, Any]:
        """Classify image using CLIP model."""
        try:
            # If CLIP not configured or disabled, fallback
            if not getattr(settings, 'enable_image_classification', True):
                return {"label": "environment", "confidence": 0.5, "method": "heuristic"}
            result = model_manager.classify_image_clip(artifact_path)
            return {
                "label": result["label"],
                "confidence": result["confidence"],
                "all_scores": result.get("all_scores", {}),
                "method": "CLIP",
            }
        except Exception as e:
            print(f"CLIP classification failed: {e}, falling back to heuristic")
            return self._classify_heuristic(artifact_path)

    def _classify_document_ai(self, text: str, artifact_path: Path = None) -> Dict[str, Any]:
        """Classify document using BERT model."""
        if not text or len(text.strip()) < 10:
            return {
                "label": "general_document",
                "confidence": 0.3,
                "method": "fallback",
            }

        try:
            result = model_manager.classify_document_bert(text)
            return {
                "label": result["label"],
                "confidence": result["confidence"],
                "method": "BERT",
            }
        except Exception as e:
            print(f"BERT classification failed: {e}, falling back to heuristic")
            return {
                "label": "general_document",
                "confidence": 0.5,
                "method": "fallback",
            }

    def _quick_text_extract(self, artifact_path: Path) -> str:
        """Quick text extraction for classification purposes."""
        try:
            if artifact_path.suffix.lower() == ".pdf":
                import pdfplumber
                with pdfplumber.open(artifact_path) as pdf:
                    text = ""
                    for page in pdf.pages[:3]:  # First 3 pages only
                        text += (page.extract_text() or "") + "\n"
                    return text
            else:
                return artifact_path.read_text(encoding="utf-8", errors="ignore")[:2000]
        except Exception:
            return ""

    def _classify_heuristic(self, artifact_path: Path) -> Dict[str, Any]:
        """Fallback heuristic-based classification."""
        name = artifact_path.name.lower()
        extension = artifact_path.suffix.lower()

        if extension in {".jpg", ".jpeg", ".png"}:
            if "cctv" in name or "cam" in name:
                label = "cctv"
            elif "injury" in name or "wound" in name or "medical" in name:
                label = "injury"
            elif "weapon" in name or "gun" in name or "knife" in name:
                label = "weapon"
            elif "scene" in name or "crime" in name:
                label = "crime_scene"
            else:
                label = "environment"
            return {"label": label, "confidence": 0.7, "method": "heuristic"}
        else:
            if "witness" in name:
                label = "witness_statement"
            elif "med" in name or "injury" in name or "hospital" in name:
                label = "medical_report"
            elif "fir" in name or "chargesheet" in name:
                label = "fir"
            elif "memo" in name or "police" in name:
                label = "police_memo"
            else:
                label = "general_document"
            return {"label": label, "confidence": 0.65, "method": "heuristic"}
