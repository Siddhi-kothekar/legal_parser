import os
from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "AI Crime Evidence Organizer"
    storage_root: Path = Field(default=Path("storage"))
    upload_dir: Path = Field(default=Path("storage/uploads"))
    processed_dir: Path = Field(default=Path("storage/processed"))
    reports_dir: Path = Field(default=Path("storage/reports"))
    case_data_dir: Path = Field(default=Path("storage/case_data"))
    
    # AI Model Settings
    enable_real_ai: bool = Field(default=True, description="Use real AI models instead of heuristics")
    enable_object_detection: bool = Field(default=True, description="Run YOLO object detection on images")
    enable_image_classification: bool = Field(default=True, description="Run CLIP image classification on images")
    openai_api_key: str = Field(default="", description="OpenAI API key for LLM reasoning (can also use OPENAI_API_KEY env var)")
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model to use")
    
    # Model paths (optional, for custom fine-tuned models)
    clip_model_path: str = Field(default="openai/clip-vit-base-patch32")
    bert_model_path: str = Field(default="nlpaueb/legal-bert-base-uncased")
    yolo_model_path: str = Field(default="yolov8n.pt")
    ner_model_path: str = Field(default="dslim/bert-base-NER")
    sentence_transformer_model_path: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    
    # Device settings
    device: str = Field(default="auto", description="Device: 'cuda', 'cpu', or 'auto'")

    model_config = {
        "env_prefix": "EVIDENCE_",
        "env_file": ".env",
        "case_sensitive": False,
    }
    
    @field_validator("openai_api_key", mode="before")
    @classmethod
    def load_openai_key(cls, v):
        """Load OpenAI API key from environment if not set in config."""
        # If already set, use it
        if v and v != "" and v != "your_openai_api_key_here":
            return v
        # Try to get from environment (supports both OPENAI_API_KEY and EVIDENCE_OPENAI_API_KEY)
        env_key = os.getenv("OPENAI_API_KEY") or os.getenv("EVIDENCE_OPENAI_API_KEY")
        if env_key:
            return env_key
        return v


settings = Settings()
settings.storage_root.mkdir(parents=True, exist_ok=True)
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.processed_dir.mkdir(parents=True, exist_ok=True)
settings.reports_dir.mkdir(parents=True, exist_ok=True)
settings.case_data_dir.mkdir(parents=True, exist_ok=True)

