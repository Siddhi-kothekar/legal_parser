from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class EvidenceIngestResponse(BaseModel):
    case_id: str
    filename: str
    stored_path: str
    message: str


class EvidenceProcessingRequest(BaseModel):
    case_id: str = Field(..., description="Case identifier returned by upload API.")
    artifact_path: str = Field(..., description="Server path of previously uploaded file.")


class EvidenceProcessingResponse(BaseModel):
    case_id: str
    artifact_path: str
    status: Literal["queued", "running", "completed", "failed"]
    message: str


class FolderUploadResponse(BaseModel):
    case_id: str
    filenames: list[str]
    stored_paths: list[str]
    message: str


class CaseProcessingRequest(BaseModel):
    case_id: str = Field(..., description="Case identifier for folder-based processing.")
    artifact_paths: list[str] | None = Field(
        default=None, description="Optional list of artifact paths. If omitted, all artifacts under case raw_files are processed."
    )


class CaseProcessingResponse(BaseModel):
    case_id: str
    artifacts_processed: int
    status: Literal["queued", "running", "completed", "failed"]
    message: str


class CaseFilesResponse(BaseModel):
    case_id: str
    files: list[str]


class ReportSummary(BaseModel):
    case_id: str
    generated_at: datetime
    timeline_events: int
    inconsistencies: int
    missing_evidence: list[str]
    report_path: str
    preview: dict[str, Any]


class CaseGraphNode(BaseModel):
    id: str
    label: str
    type: Literal["person", "location"]
    cases: list[str]
    count: int


class CaseGraphEdge(BaseModel):
    id: str
    source: str
    target: str
    weight: int
    cases: list[str]
    case_details: Optional[list[dict[str, Any]]] = None
    person: str
    location: str


class CaseGraphResponse(BaseModel):
    generated_at: datetime
    nodes: list[CaseGraphNode]
    edges: list[CaseGraphEdge]

