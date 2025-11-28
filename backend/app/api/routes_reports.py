import json
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.models.schemas import CaseGraphResponse, ReportSummary
from app.services.case_graph import CaseGraphService


router = APIRouter()
case_graph_service = CaseGraphService()


@router.get("/graph", response_model=CaseGraphResponse)
async def get_case_graph() -> CaseGraphResponse:
    graph = case_graph_service.build_graph_payload()
    return CaseGraphResponse(**graph)


@router.get("/{case_id}", response_model=ReportSummary)
async def get_report(case_id: str) -> ReportSummary:
    # Check if processing is in progress
    status_path = settings.reports_dir / f"{case_id}.status"
    if status_path.exists():
        status = status_path.read_text(encoding="utf-8").strip()
        if status == "processing":
            # Return 202 Accepted to indicate processing in progress
            raise HTTPException(
                status_code=202,
                detail="Report is being processed. Please wait and try again in a few seconds."
            )
        elif status == "failed":
            raise HTTPException(
                status_code=500,
                detail="Report processing failed. Please check server logs."
            )
    
    # Try to get detailed report first
    detailed_path = settings.reports_dir / f"{case_id}.json"
    summary_path = settings.reports_dir / f"{case_id}_summary.json"
    
    if detailed_path.exists():
        # Return detailed report with all extracted content
        try:
            data = json.loads(detailed_path.read_text(encoding="utf-8"))
            # Convert to ReportSummary format
            report_summary = ReportSummary(
                case_id=data.get("case_id", case_id),  # Fallback to case_id from URL
                generated_at=datetime.fromisoformat(data.get("generated_at", datetime.utcnow().isoformat())),
                timeline_events=len(data.get("timeline_events", [])),
                inconsistencies=len(data.get("inconsistencies", [])),
                missing_evidence=data.get("missing_evidence", []),
                report_path=str(settings.reports_dir / f"{case_id}.pdf"),
                preview={
                    "timeline": [e.get("description", str(e)) for e in data.get("timeline_events", [])[:5]],
                    "inconsistencies": data.get("inconsistencies", []),
                    "extracted_content": data.get("extracted_content", []),
                    "evidence_map": data.get("evidence_map", {}),
                    "case_summary": data.get("case_summary", ""),
                "relationships": data.get("relationships", []),
                },
            )
            print(f"[Reports API] Returning report for {case_id}: case_id={report_summary.case_id}, timeline_events={report_summary.timeline_events}, inconsistencies={report_summary.inconsistencies}")
            return report_summary
        except Exception as e:
            print(f"Error loading detailed report for {case_id}: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Error loading report: {str(e)}"
            )
    elif summary_path.exists():
        # Fallback to summary if detailed doesn't exist
        try:
            data = summary_path.read_text(encoding="utf-8")
            summary = ReportSummary.model_validate_json(data)
            print(f"[Reports API] Returning summary for {case_id}: case_id={summary.case_id}, timeline_events={summary.timeline_events}")
            return summary
        except Exception as e:
            print(f"Error loading summary report for {case_id}: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Error loading summary report: {str(e)}"
            )
    else:
        # No status file and no report - likely not started yet
        raise HTTPException(
            status_code=202,  # Use 202 instead of 404 to indicate "accepted but not ready"
            detail="Report not ready yet. Processing may still be in progress (models downloading on first run). Please wait and try again in a few seconds."
        )


@router.get("/{case_id}/download")
async def download_pdf(case_id: str) -> dict[str, str]:
    pdf_path = settings.reports_dir / f"{case_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF not generated.")
    return {"case_id": case_id, "pdf_path": str(Path(pdf_path).resolve())}

