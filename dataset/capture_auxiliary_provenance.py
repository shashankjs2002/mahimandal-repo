"""Record auxiliary-data provenance without executing the collection notebook.

Current provider metadata is distinguished from missing historical tile lineage.
Never upgrades a live service snapshot into proof of the original raster version.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from audit_sentinel_product_metadata import EXPECTED_MANIFEST, digest, fetch, write_json


def notebook_evidence(path: Path) -> dict:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    constants, functions = {}, []
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        raw = cell.get("source", [])
        source = raw if isinstance(raw, str) else "".join(raw)
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and (target.id.startswith("ESRI_") or target.id == "MAX_ESRI_CLOUD_PCT"):
                        try:
                            constants[target.id] = ast.literal_eval(node.value)
                        except (ValueError, TypeError):
                            pass
            if isinstance(node, ast.FunctionDef) and node.name in (
                "classify_esri_landscape", "classify_landscape", "read_esri_landcover_2025"):
                snippet = ast.get_source_segment(source, node)
                functions.append({"name": node.name, "cell_index": index,
                                  "source_sha256": digest(snippet.encode()), "source": snippet})
    return {"notebook_sha256": digest(path.read_bytes()), "constants": constants,
            "classification_and_query_source": functions,
            "method": "AST extraction from code cells; notebook outputs ignored; no code executed"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--notebook", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if digest(args.manifest.read_bytes()) != EXPECTED_MANIFEST:
        raise ValueError("Not the frozen release manifest")
    with args.manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    evidence = notebook_evidence(args.notebook)
    service = evidence["constants"]["ESRI_LANDCOVER_SERVICE"]
    if service != "https://ic.imagery1.arcgis.com/arcgis/rest/services/Sentinel2_10m_LandCover/ImageServer":
        raise ValueError("Unexpected Esri service in source notebook")
    args.output.mkdir(parents=True, exist_ok=True)
    providers = []
    for name, url in (
        ("esri_service.json", service + "?f=pjson"),
        ("copernicus_dem_collection.json", "https://planetarycomputer.microsoft.com/api/stac/v1/collections/cop-dem-glo-30"),
    ):
        record = {"url": url, "file": name,
                  "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                  "scope": "Current provider metadata, not an archived collection-time response"}
        try:
            raw = fetch(url)
            document = json.loads(raw)
            if "error" in document:
                raise ValueError("Provider returned a JSON error")
            (args.output / name).write_bytes(raw)
            record.update(status="retrieved", sha256=digest(raw))
        except Exception as exc:
            record.update(status="unresolved", error=str(exc))
        providers.append(record)
    observed = {column: dict(Counter(row[column] for row in rows))
                for column in ("landcover_year", "landcover_source", "landcover_service", "worldcover_year")
                if column in rows[0]}
    result = {"manifest_sha256": EXPECTED_MANIFEST, "pair_count": len(rows),
              "script_sha256": digest(Path(__file__).read_bytes()),
              "original_notebook": evidence, "manifest_auxiliary_fields": observed,
              "current_provider_metadata": providers,
              "historical_lineage_status": "partial",
              "remaining": [
                  "Original per-patch Esri contributing raster IDs and service response snapshots were not established by this audit.",
                  "Original per-patch DEM source item IDs/version receipts were not established by this audit.",
                  "A live metadata response cannot prove the exact raster bytes returned during original preparation.",
              ]}
    write_json(args.output / "provenance.json", result)
    print(json.dumps({"pair_count": len(rows), "provider_status": providers,
                      "historical_lineage_status": "partial"}, indent=2))


if __name__ == "__main__":
    main()
