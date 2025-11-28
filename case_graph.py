from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from dateutil import parser as date_parser

from app.config import settings
from app.services.model_manager import model_manager


class CaseGraphService:
    """Builds and maintains relationships between processed cases."""

    def __init__(self) -> None:
        self.reports_dir: Path = settings.reports_dir

    # ------------------------------------------------------------------
    # Public APIs
    # ------------------------------------------------------------------
    def compute_relationships_for_case(self, case_id: str, case_data: dict) -> List[dict]:
        """Compute relationships between the provided case and all existing cases."""
        current_facts = self._extract_case_facts(case_data)
        other_cases = self._load_existing_case_facts(exclude_case_id=case_id)
        relationships: List[dict] = []

        for other_id, other_facts in other_cases.items():
            link = self._compare_case_facts(case_id, current_facts, other_id, other_facts)
            if link:
                relationships.append(link)

        # Sort deterministic for UI
        relationships.sort(key=lambda x: (-x.get("weight", 0), x.get("target")))
        return relationships

    def refresh_relationship_references(self, case_id: str, relationships: List[dict]) -> None:
        """
        Ensure every other case file reflects the latest relationship status with `case_id`.
        Removes stale references, then adds mirrored entries for current relationships.
        """
        self._remove_relationship_from_all(case_id)
        for link in relationships:
            target_id = link.get("target")
            if not target_id:
                continue
            mirrored = {
                "source": target_id,
                "target": case_id,
                "reasons": link.get("reasons", []),
                "weight": link.get("weight"),
                "similarity": link.get("similarity"),
                "shared_entities": link.get("shared_entities", {}),
            }
            self._upsert_relationship_entry(target_id, mirrored)

    def build_graph_payload(self) -> dict:
        """
        Return a graph payload with person/location nodes and edges representing shared cases.
        """
        case_facts = list(self._load_existing_case_facts().items())
        return self._build_entity_graph(case_facts)

    # ------------------------------------------------------------------
    # Case fact loading / extraction
    # ------------------------------------------------------------------
    def _load_existing_case_facts(self, exclude_case_id: Optional[str] = None) -> Dict[str, dict]:
        cases: Dict[str, dict] = {}
        for path in self.reports_dir.glob("*.json"):
            if path.name.endswith("_summary.json"):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            case_id = data.get("case_id") or path.stem.replace("_summary", "")
            if exclude_case_id and case_id == exclude_case_id:
                continue
            cases[case_id] = self._extract_case_facts(data)
        return cases

    def _extract_case_facts(self, case_data: dict) -> dict:
        evidence_map = case_data.get("evidence_map") or {}
        extracted_content = case_data.get("extracted_content") or []

        persons, person_labels = self._canonicalize_entities(evidence_map.get("case_persons"))
        locations, location_labels = self._canonicalize_entities(evidence_map.get("case_locations"))
        timestamps = set(evidence_map.get("case_timestamps") or [])
        weapons, weapon_labels = self._canonicalize_entities(evidence_map.get("case_weapons"))
        injuries, injury_labels = self._canonicalize_entities(evidence_map.get("case_injuries"))

        event_types: Set[str] = set()
        for content in extracted_content:
            for event in content.get("events") or []:
                if isinstance(event, dict):
                    if event.get("type"):
                        event_types.add(event["type"])
                elif isinstance(event, str):
                    event_types.add(event)

        summary_text = case_data.get("case_summary") or evidence_map.get("consolidated_text", "")
        evidence_count = len(extracted_content)
        files = evidence_map.get("files", [])

        return {
            "case_id": case_data.get("case_id"),
            "persons": persons,
            "person_labels": person_labels,
            "locations": locations,
            "location_labels": location_labels,
            "timestamps": timestamps,
            "dates": self._normalize_dates(timestamps),
            "weapons": weapons,
            "weapon_labels": weapon_labels,
            "injuries": injuries,
            "injury_labels": injury_labels,
            "events": event_types,
            "summary_text": summary_text,
            "evidence_count": evidence_count,
            "files": files,
        }

    # ------------------------------------------------------------------
    # Relationship computation
    # ------------------------------------------------------------------
    def _compare_case_facts(
        self,
        source_id: str,
        source: dict,
        target_id: str,
        target: dict,
    ) -> Optional[dict]:
        reasons: List[str] = []
        shared: Dict[str, List[str]] = {}

        shared_person_keys = sorted(source["persons"] & target["persons"])
        if shared_person_keys:
            shared_persons = self._display_entities(shared_person_keys, source.get("person_labels", {}), target.get("person_labels", {}))
            reasons.append(f"👤 Same person: {', '.join(shared_persons[:5])}")
            shared["persons"] = shared_persons

        shared_location_keys = sorted(source["locations"] & target["locations"])
        if shared_location_keys:
            shared_locations = self._display_entities(shared_location_keys, source.get("location_labels", {}), target.get("location_labels", {}))
            reasons.append(f"📍 Same location: {', '.join(shared_locations[:5])}")
            shared["locations"] = shared_locations

        shared_weapon_keys = sorted(source["weapons"] & target["weapons"])
        if shared_weapon_keys:
            shared_weapons = self._display_entities(shared_weapon_keys, source.get("weapon_labels", {}), target.get("weapon_labels", {}))
            reasons.append(f"🗡️ Same weapon type: {', '.join(shared_weapons[:3])}")
            shared["weapons"] = shared_weapons

        shared_injury_keys = sorted(source["injuries"] & target["injuries"])
        if shared_injury_keys:
            shared_injuries = self._display_entities(shared_injury_keys, source.get("injury_labels", {}), target.get("injury_labels", {}))
            reasons.append(f"🩺 Similar injuries: {', '.join(shared_injuries[:3])}")
            shared["injuries"] = shared_injuries

        shared_events = sorted(source["events"] & target["events"])
        if shared_events:
            reasons.append(f"⚡ Matching event types: {', '.join(shared_events[:3])}")
            shared["events"] = shared_events

        overlapping_dates = self._dates_within_window(source["dates"], target["dates"])
        if overlapping_dates:
            reasons.append("🕒 Incidents within 48 hours")
            shared["dates"] = [d.isoformat() for d in overlapping_dates[:3]]

        similarity = self._semantic_similarity(source.get("summary_text"), target.get("summary_text"))
        if similarity >= 0.70:
            reasons.append(f"🤖 Narrative similarity {similarity:.2f}")

        if not reasons:
            return None

        weight = len(reasons) + (similarity if similarity > 0 else 0)
        
        # Only return relationship if it meets minimum threshold
        # Require at least 2 strong indicators OR high semantic similarity
        min_weight = 2.0
        if weight < min_weight and similarity < 0.75:
            return None

        return {
            "source": source_id,
            "target": target_id,
            "reasons": reasons,
            "weight": round(weight, 2),
            "similarity": round(similarity, 3) if similarity else 0.0,
            "shared_entities": shared,
        }

    # ------------------------------------------------------------------
    # Graph helpers
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # File mutation helpers
    # ------------------------------------------------------------------
    def _upsert_relationship_entry(self, case_id: str, relationship: dict) -> None:
        path = self._case_report_path(case_id)
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return

        relationships = data.get("relationships") or []
        target = relationship.get("target")
        relationships = [rel for rel in relationships if rel.get("target") != target]
        relationships.append(relationship)
        relationships.sort(key=lambda x: (-x.get("weight", 0), x.get("target")))
        data["relationships"] = relationships

        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def _remove_relationship_from_all(self, target_case_id: str) -> None:
        """Remove references to target_case_id from every other case file."""
        for path in self.reports_dir.glob("*.json"):
            if path.name.endswith("_summary.json"):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            relationships = data.get("relationships")
            if not relationships:
                continue
            filtered = [rel for rel in relationships if rel.get("target") != target_case_id]
            if len(filtered) == len(relationships):
                continue
            data["relationships"] = filtered
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def _case_report_path(self, case_id: str) -> Path:
        return self.reports_dir / f"{case_id}.json"

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _normalize_dates(self, timestamps: Set[str]) -> Set[date]:
        values: Set[date] = set()
        for raw in timestamps:
            parsed = self._safe_parse_date(raw)
            if parsed:
                values.add(parsed)
        return values

    def _dates_within_window(self, dates_a: Set[date], dates_b: Set[date], window_days: int = 2) -> List[date]:
        overlaps: Set[date] = set()
        for d_a in dates_a:
            for d_b in dates_b:
                if abs((d_a - d_b).days) <= window_days:
                    overlaps.add(min(d_a, d_b))
        return sorted(overlaps)

    def _safe_parse_date(self, raw: str) -> Optional[date]:
        try:
            parsed = date_parser.parse(raw)
            return parsed.date()
        except Exception:
            return None

    def _semantic_similarity(self, text_a: Optional[str], text_b: Optional[str]) -> float:
        if not text_a or not text_b:
            return 0.0
        return model_manager.compute_text_similarity(text_a, text_b)

    # ------------------------------------------------------------------
    # Entity graph builders
    # ------------------------------------------------------------------
    def _build_entity_graph(self, case_facts: List[Tuple[str, dict]]) -> dict:
        person_cases: Dict[str, Set[str]] = defaultdict(set)
        location_cases: Dict[str, Set[str]] = defaultdict(set)
        person_labels: Dict[str, str] = {}
        location_labels: Dict[str, str] = {}
        edge_cases: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for case_id, facts in case_facts:
            persons = set(facts.get("persons") or [])
            locations = set(facts.get("locations") or [])
            person_label_map = facts.get("person_labels", {}) or {}
            location_label_map = facts.get("location_labels", {}) or {}
            summary = (facts.get("summary_text") or "")[:300]

            for key in persons:
                person_cases[key].add(case_id)
                if key not in person_labels:
                    person_labels[key] = person_label_map.get(key, key)
            for key in locations:
                location_cases[key].add(case_id)
                if key not in location_labels:
                    location_labels[key] = location_label_map.get(key, key)

            if not persons or not locations:
                continue

            for person in persons:
                for location in locations:
                    if not person or not location:
                        continue
                    link_key = (person, location)
                    entry = edge_cases.setdefault(
                        link_key,
                        {
                            "cases": set(),
                            "details": [],
                        },
                    )
                    if case_id not in entry["cases"]:
                        entry["cases"].add(case_id)
                        if summary:
                            entry["details"].append(
                                {
                                    "case_id": case_id,
                                    "summary": summary,
                                }
                            )

        nodes = []
        for key, cases in sorted(person_cases.items(), key=lambda x: x[0]):
            nodes.append(
                {
                    "id": self._entity_node_id("person", key),
                    "label": person_labels.get(key, key),
                    "type": "person",
                    "cases": sorted(cases),
                    "count": len(cases),
                }
            )

        for key, cases in sorted(location_cases.items(), key=lambda x: x[0]):
            nodes.append(
                {
                    "id": self._entity_node_id("location", key),
                    "label": location_labels.get(key, key),
                    "type": "location",
                    "cases": sorted(cases),
                    "count": len(cases),
                }
            )

        edges = []
        for (person, location), payload in edge_cases.items():
            edge_id = self._edge_id(person, location)
            edges.append(
                {
                    "id": edge_id,
                    "source": self._entity_node_id("person", person),
                    "target": self._entity_node_id("location", location),
                    "weight": len(payload["cases"]),
                    "cases": sorted(payload["cases"]),
                    "case_details": payload["details"][:10],
                    "person": person_labels.get(person, person),
                    "location": location_labels.get(location, location),
                }
            )

        return {
            "generated_at": datetime.utcnow().isoformat(),
            "nodes": nodes,
            "edges": edges,
        }

    def _entity_node_id(self, kind: str, value: str) -> str:
        slug = self._slugify(value)
        return f"{kind}:{slug}"

    def _edge_id(self, person: str, location: str) -> str:
        return f"edge:{self._slugify(person)}:{self._slugify(location)}"

    def _slugify(self, value: str) -> str:
        if not value:
            return "unknown"
        slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
        return slug or "unknown"

    def _canonicalize_entities(self, values: Optional[List[str]]) -> Tuple[Set[str], Dict[str, str]]:
        keys: Set[str] = set()
        labels: Dict[str, str] = {}
        if not values:
            return keys, labels
        for raw in values:
            cleaned = self._clean_value(raw)
            if not cleaned:
                continue
            key = cleaned.lower()
            keys.add(key)
            labels.setdefault(key, cleaned)
        return keys, labels

    def _display_entities(self, keys: List[str], primary_map: Dict[str, str], secondary_map: Dict[str, str]) -> List[str]:
        display = []
        for key in keys:
            label = primary_map.get(key) or secondary_map.get(key) or key
            display.append(label)
        return display

    def _clean_value(self, value: Optional[str]) -> str:
        if not value:
            return ""
        return re.sub(r"\s+", " ", value).strip()

