from pathlib import Path

from app.services.classification import ClassificationService
from app.services.extraction import ExtractionService
from app.services.ingestion import IngestionService
from app.services.normalization import NormalizationService
from app.services.reasoning import ReasoningService
from app.services.reporting import ReportingService
from app.services.timeline import TimelineService


class EvidencePipeline:
    """
    High-level orchestrator implementing the full methodology:

    1. File type identification (IngestionService)
    2. Evidence classification (ClassificationService)
    3. Content extraction (ExtractionService)
    4. Data normalization (NormalizationService)
    5. Timeline construction (TimelineService)
    6. Inconsistency detection & missing evidence suggestion (ReasoningService)
    7. Report generation (ReportingService)
    """

    def __init__(self) -> None:
        self.ingestion = IngestionService()
        self.classification = ClassificationService()
        self.extraction = ExtractionService()
        self.normalization = NormalizationService()
        self.timeline = TimelineService()
        self.reasoning = ReasoningService()
        self.reporting = ReportingService()

    def execute(self, case_id: str, artifact_path: Path) -> None:
        """
        Execute the full evidence processing pipeline.
        Includes error handling and logging for debugging.
        """
        try:
            print(f"[Pipeline] Starting processing for case {case_id}")
            
            # Step 2: low-level file identification
            ingest_meta = self.ingestion.validate(artifact_path)
            print(f"[Pipeline] File validated: {ingest_meta.get('high_level_type', 'unknown')}")

            # Step 3: semantic evidence classification
            classification = self.classification.classify(artifact_path)
            print(f"[Pipeline] Classified as: {classification.get('label', 'unknown')}")

            # Step 4: content extraction
            extraction = self.extraction.extract(artifact_path)
            print(f"[Pipeline] Content extracted: {extraction.get('type', 'unknown')}")

            # Step 3 (if needed): re-classify with extracted text for better accuracy
            if extraction.get("type") == "document" and "raw_text" in extraction:
                classification = self.classification.classify(
                    artifact_path,
                    extracted_text=extraction.get("raw_text", "")
                )
                print(f"[Pipeline] Re-classified with text: {classification.get('label', 'unknown')}")

            # Step 5: normalization
            normalized = self.normalization.normalize(extraction)
            normalized["classification"] = classification
            normalized["ingestion"] = ingest_meta
            normalized_data = [normalized]

            # Step 6: timeline construction
            events = self.timeline.build(case_id, normalized_data)
            print(f"[Pipeline] Timeline built: {len(events)} events")

            # Step 7/8: inconsistency detection and missing evidence suggestion
            inconsistencies = self.reasoning.detect_inconsistencies(events, normalized_data)
            missing = self.reasoning.suggest_missing_evidence(events, normalized_data)
            warnings = self.reasoning.get_warnings()
            print(f"[Pipeline] Found {len(inconsistencies)} inconsistencies, {len(missing)} missing evidence suggestions")

            # Step 9: reporting
            self.reporting.persist_summary(case_id, events, inconsistencies, missing, normalized_data, warnings)
            print(f"[Pipeline] ✅ Report saved successfully for case {case_id}")
            
        except Exception as e:
            print(f"[Pipeline] ❌ ERROR processing case {case_id}: {e}")
            import traceback
            traceback.print_exc()
            raise

    def execute_case(self, case_id: str, case_dir: Path) -> int:
        """
        Process all artifact files within a case directory's raw_files and return
        the number of processed artifacts.
        """
        processed = 0
        normalized_data = []
        for artifact in case_dir.iterdir():
            if not artifact.is_file():
                continue
            try:
                print(f"[Pipeline] Processing artifact: {artifact}")
                # Step 2: low-level file identification
                ingest_meta = self.ingestion.validate(artifact)

                # Step 3: semantic evidence classification
                classification = self.classification.classify(artifact)

                # Step 4: content extraction
                extraction = self.extraction.extract(artifact)

                # Re-classify with extracted text if document
                if extraction.get("type") == "document" and "raw_text" in extraction:
                    classification = self.classification.classify(
                        artifact_path=artifact, extracted_text=extraction.get("raw_text", "")
                    )

                normalized = self.normalization.normalize(extraction)
                normalized["filename"] = artifact.name
                normalized["classification"] = classification
                normalized["ingestion"] = ingest_meta
                normalized_data.append(normalized)
                processed += 1
            except Exception as e:
                print(f"[Pipeline] Error processing artifact {artifact}: {e}")

        # After processing all artifacts, produce a single report for the case
        try:
            events = self.timeline.build(case_id, normalized_data)
            inconsistencies = self.reasoning.detect_inconsistencies(events, normalized_data)
            missing = self.reasoning.suggest_missing_evidence(events, normalized_data)
            warnings = self.reasoning.get_warnings()
            self.reporting.persist_summary(case_id, events, inconsistencies, missing, normalized_data, warnings)
            print(f"[Pipeline] ✅ Case-level report generated for {case_id} with {processed} artifacts")
        except Exception as e:
            print(f"[Pipeline] Error generating case-level report for {case_id}: {e}")

        return processed

