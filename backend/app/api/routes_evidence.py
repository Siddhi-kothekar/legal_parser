from pathlib import Path
from uuid import uuid4
from typing import List
import zipfile

from fastapi import APIRouter, BackgroundTasks, File, UploadFile, HTTPException

from app.config import settings
from app.models.schemas import (
    EvidenceIngestResponse,
    EvidenceProcessingRequest,
    EvidenceProcessingResponse,
    FolderUploadResponse,
    CaseProcessingRequest,
    CaseProcessingResponse,
    CaseFilesResponse,
)
from app.services.pipeline import EvidencePipeline


router = APIRouter()
pipeline = EvidencePipeline()


@router.post("/upload", response_model=EvidenceIngestResponse)
async def upload_evidence(file: UploadFile = File(...)) -> EvidenceIngestResponse:
    case_id = uuid4().hex
    # Store single-file uploads in the case_data raw_files to keep organization
    case_dir = settings.case_data_dir / case_id / "raw_files"
    case_dir.mkdir(parents=True, exist_ok=True)
    target_path = case_dir / file.filename
    content = await file.read()
    target_path.write_bytes(content)
    # Validate file type and basic checks
    ingest_meta = pipeline.ingestion.validate(target_path)
    return EvidenceIngestResponse(
        case_id=case_id,
        filename=file.filename,
        stored_path=str(target_path),
        message=f"Evidence stored ({ingest_meta.get('high_level_type')}); Trigger processing via /evidence/process.",
    )

@router.post("/upload_folder", response_model=FolderUploadResponse)
async def upload_evidence_folder(files: List[UploadFile] = File(...)) -> FolderUploadResponse:
    """
    Accept multiple files or a single zip archive containing a folder of evidence.
    Files are stored under `storage/case_data/{case_id}/raw_files/`.
    """
    case_id = uuid4().hex
    from app.config import settings

    case_dir = settings.case_data_dir / case_id / "raw_files"
    case_dir.mkdir(parents=True, exist_ok=True)

    stored_paths = []
    filenames = []
    # If client uploaded a single zip, extract it
    if len(files) == 1 and files[0].filename.lower().endswith(".zip"):
        file = files[0]
        # Save the zip temporarily then extract
        zip_path = settings.upload_dir / f"{case_id}_{file.filename}"
        content = await file.read()
        zip_path.write_bytes(content)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(case_dir)
        zip_path.unlink()
        # Collect the extracted files
        for p in case_dir.rglob("*"):
            if p.is_file():
                filenames.append(p.name)
                stored_paths.append(str(p))
                # Validate each extracted file
                _ = pipeline.ingestion.validate(p)
    else:
        for file in files:
            target_path = case_dir / file.filename
            content = await file.read()
            target_path.write_bytes(content)
            filenames.append(file.filename)
            stored_paths.append(str(target_path))
            _ = pipeline.ingestion.validate(target_path)

    return FolderUploadResponse(
        case_id=case_id,
        filenames=filenames,
        stored_paths=stored_paths,
        message=f"Stored {len(stored_paths)} files for case {case_id} under raw_files.",
    )


@router.post("/process", response_model=EvidenceProcessingResponse)
async def process_evidence(
    payload: EvidenceProcessingRequest, tasks: BackgroundTasks
) -> EvidenceProcessingResponse:
    """
    Process evidence in background.
    Note: First-time processing may take 2-5 minutes due to model downloads.
    The report will be available at /reports/{case_id} when ready.
    """
    artifact_path = Path(payload.artifact_path)
    if not artifact_path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found on server.")

    def _run_pipeline() -> None:
        """Background task to run the pipeline."""
        try:
            # Create status file to indicate processing has started
            from app.config import settings
            status_path = settings.reports_dir / f"{payload.case_id}.status"
            status_path.write_text("processing", encoding="utf-8")
            
            print(f"[Background Task] Starting pipeline for case {payload.case_id}")
            pipeline.execute(case_id=payload.case_id, artifact_path=artifact_path)
            print(f"[Background Task] Pipeline completed for case {payload.case_id}")
            
            # Remove status file when done (report file will exist)
            if status_path.exists():
                status_path.unlink()
        except Exception as e:
            print(f"[Background Task] ERROR in pipeline for case {payload.case_id}: {e}")
            import traceback
            traceback.print_exc()
            # Mark as failed
            from app.config import settings
            status_path = settings.reports_dir / f"{payload.case_id}.status"
            status_path.write_text("failed", encoding="utf-8")

    tasks.add_task(_run_pipeline)
    return EvidenceProcessingResponse(
        case_id=payload.case_id,
        artifact_path=str(artifact_path),
        status="queued",
        message="Processing started in background. This may take 2-5 minutes on first run (models downloading). Poll /reports/{case_id} to check status.",
    )

@router.post("/process_case", response_model=CaseProcessingResponse)
async def process_case(payload: CaseProcessingRequest, tasks: BackgroundTasks) -> CaseProcessingResponse:
    """
    Process an entire case folder. If `artifact_paths` is omitted, all files under
    `storage/case_data/{case_id}/raw_files/` will be processed.
    """
    from app.config import settings

    case_dir = settings.case_data_dir / payload.case_id / "raw_files"
    if not case_dir.exists():
        raise HTTPException(status_code=404, detail="Case folder not found on server.")

    if payload.artifact_paths:
        artifacts = [Path(p) for p in payload.artifact_paths]
    else:
        artifacts = [p for p in case_dir.iterdir() if p.is_file()]

    if not artifacts:
        raise HTTPException(status_code=400, detail="No artifacts found to process for this case.")

    def _run_case_pipeline() -> None:
        try:
            from app.config import settings
            status_path = settings.reports_dir / f"{payload.case_id}.status"
            status_path.write_text("processing", encoding="utf-8")
            print(f"[Background Task] Starting pipeline for case folder {payload.case_id}")
            processed_count = pipeline.execute_case(case_id=payload.case_id, case_dir=case_dir)
            print(f"[Background Task] Processed {processed_count} artifacts for case {payload.case_id}")

            # Remove status file when done
            if status_path.exists():
                status_path.unlink()
            print(f"[Background Task] Pipeline completed for case folder {payload.case_id}")
        except Exception as e:
            print(f"[Background Task] ERROR in pipeline for case folder {payload.case_id}: {e}")
            import traceback
            traceback.print_exc()
            status_path = settings.reports_dir / f"{payload.case_id}.status"
            status_path.write_text("failed", encoding="utf-8")

    tasks.add_task(_run_case_pipeline)
    return CaseProcessingResponse(
        case_id=payload.case_id,
        artifacts_processed=len(artifacts),
        status="queued",
        message=f"Processing queued for case {payload.case_id}. {len(artifacts)} artifacts will be processed.",
    )


@router.get("/cases/{case_id}/files", response_model=CaseFilesResponse)
async def list_case_files(case_id: str) -> CaseFilesResponse:
    from app.config import settings
    case_dir = settings.case_data_dir / case_id / "raw_files"
    if not case_dir.exists():
        raise HTTPException(status_code=404, detail="Case not found")
    files = [str(p.name) for p in case_dir.iterdir() if p.is_file()]
    return CaseFilesResponse(case_id=case_id, files=files)
