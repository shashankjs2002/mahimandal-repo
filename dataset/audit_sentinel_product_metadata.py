"""Independently compare Sentinel product XML radiometry with release conversion.

Metadata only: no raster download, model execution, or dataset modification.
Uses Python's standard library. Cached STAC/XML bytes and hashes support offline
reproduction. Product metadata consistency is not a full-pixel radiometric audit.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from threading import Lock
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


EXPECTED_MANIFEST = "6bf97403ee40ccf0cbd38b2e6e4932e01973e5d00e211e0acfb1e0b810efc5e7"
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
TOKEN = "https://planetarycomputer.microsoft.com/api/sas/v1/token"
TOKEN_CACHE = {}
TOKEN_LOCK = Lock()
BANDS = ("B02", "B03", "B04", "B08")
REFERENCES = [
    "https://sentiwiki.copernicus.eu/web/s2-products",
    "https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a#Baseline-Change",
]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def write_json(path: Path, value) -> None:
    atomic_write(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def unsigned(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def fetch(url: str) -> bytes:
    for attempt in range(3):
        try:
            request = Request(url, headers={"User-Agent": "MahimandalSR-metadata-review/1.0"})
            with urlopen(request, timeout=45) as response:
                content = response.read(8 * 1024 * 1024 + 1)
            if len(content) > 8 * 1024 * 1024:
                raise ValueError("Unexpectedly large metadata response")
            return content
        except HTTPError as exc:
            if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == 2:
                raise RuntimeError(f"Metadata HTTP {exc.code} at {unsigned(url)}") from None
            retry_after = exc.headers.get("Retry-After", "")
            if retry_after.isdigit():
                time.sleep(min(int(retry_after), 60))
        except (URLError, TimeoutError):
            if attempt == 2:
                raise RuntimeError(f"Metadata connection failed at {unsigned(url)}") from None
        time.sleep(2 ** (attempt + 1))
    raise RuntimeError("Metadata retries exhausted")


def signed_metadata_url(href: str) -> str:
    """Reuse a container-scoped token, avoiding per-product signing requests."""
    parts = urlsplit(href)
    account = (parts.hostname or "").split(".", 1)[0]
    container = parts.path.strip("/").split("/", 1)[0]
    key = (account, container)
    with TOKEN_LOCK:
        token = TOKEN_CACHE.get(key)
        if token is None or token[1] - time.time() < 120:
            record = json.loads(fetch(f"{TOKEN}/{quote(account, safe='')}/{quote(container, safe='')}"))
            expiry = datetime.fromisoformat(record["msft:expiry"].replace("Z", "+00:00")).timestamp()
            if expiry - time.time() < 120:
                raise ValueError("Provider returned an expired or nearly expired read token")
            token = (record["token"], expiry)
            TOKEN_CACHE[key] = token
    return href + "?" + token[0]


def local_name(element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def one_text(root, name: str) -> str:
    values = [(node.text or "").strip() for node in root.iter() if local_name(node) == name]
    if len(values) != 1 or not values[0]:
        raise ValueError(f"Expected exactly one nonempty XML {name}")
    return values[0]


def canonical_band(name: str) -> str:
    name = name.upper()
    return "B" + name[1:].zfill(2) if name.startswith("B") and name[1:].isdigit() else name


def parse_product(xml: bytes) -> dict:
    if b"<!DOCTYPE" in xml.upper() or b"<!ENTITY" in xml.upper():
        raise ValueError("XML DTD/entities are not allowed")
    root = ET.fromstring(xml)
    product = one_text(root, "PRODUCT_URI")
    baseline_text = one_text(root, "PROCESSING_BASELINE")
    baseline = float(baseline_text)
    quantification = float(one_text(root, "BOA_QUANTIFICATION_VALUE"))
    if not math.isfinite(baseline) or baseline < 0:
        raise ValueError("Invalid XML processing baseline")
    if not math.isfinite(quantification) or quantification <= 0:
        raise ValueError("Invalid BOA quantification value")
    identifiers = {}
    for node in root.iter():
        if local_name(node) == "Spectral_Information":
            identifier = node.attrib["bandId"]
            if identifier in identifiers:
                raise ValueError("Duplicate spectral band identifier")
            identifiers[identifier] = canonical_band(node.attrib["physicalBand"])
    offsets = {}
    for node in root.iter():
        if local_name(node) == "BOA_ADD_OFFSET":
            identifier = node.attrib["band_id"]
            if identifier not in identifiers:
                raise ValueError("Offset has no explicit XML physical-band mapping")
            band = identifiers[identifier]
            value = float((node.text or "").strip())
            if band in offsets or not math.isfinite(value):
                raise ValueError("Duplicate or invalid BOA offset")
            offsets[band] = value
    missing = set(BANDS) - set(offsets)
    if missing:
        if baseline >= 4.0 or offsets:
            raise ValueError(f"Required explicit offsets missing: {sorted(missing)}")
        # The pre-PB04 zero-offset convention is documented, not a post-PB04 fallback.
        if not set(BANDS).issubset(identifiers.values()):
            raise ValueError("Missing physical-band definitions")
        offsets = {band: 0.0 for band in BANDS}
    return {"product_uri": product, "processing_baseline": baseline_text,
            "boa_quantification_value": quantification,
            "boa_add_offsets": {band: offsets[band] for band in BANDS},
            "offset_evidence": "explicit_product_xml" if baseline >= 4.0 or not missing
            else "documented_pre_pb04_zero_offset"}


def assess(item: dict, xml: bytes, identifier: str) -> dict:
    if item.get("id") != identifier or item.get("collection") != "sentinel-2-l2a":
        raise ValueError("STAC identity/collection mismatch")
    parsed = parse_product(xml)
    properties = item["properties"]
    if parsed["product_uri"] != properties.get("s2:product_uri"):
        raise ValueError("XML and STAC product identifiers differ")
    baseline = float(properties["s2:processing_baseline"])
    if baseline != float(parsed["processing_baseline"]):
        raise ValueError("XML and STAC processing baselines differ")
    notebook_offset = -1000.0 if baseline >= 4.0 else 0.0
    quantification = parsed["boa_quantification_value"]
    maximum_difference = max(
        abs((dn + parsed["boa_add_offsets"][band]) / quantification
            - (dn + notebook_offset) / 10000.0)
        for band in BANDS for dn in (0, 65535)
    )
    agrees = maximum_difference < 1e-12
    return {"sentinel_id": identifier, **parsed,
            "notebook_additive_offset_dn": notebook_offset,
            "notebook_quantification_value": 10000.0,
            "max_unclipped_formula_difference_uint16": maximum_difference,
            "status": "metadata_supports_conversion" if agrees else "conversion_disagrees_with_metadata",
            "band_asset_raster_metadata": {
                band: item["assets"][band].get("raster:bands", []) for band in BANDS},
            "scope": "Product XML coefficients and identity; not an all-band pixel audit or registration validation"}


def cached_source(identifier: str, cache: Path, offline: bool) -> tuple[dict, bytes, dict]:
    directory = cache / digest(identifier.encode("utf-8"))[:24]
    directory.mkdir(parents=True, exist_ok=True)
    stac_path, xml_path, receipt_path = [directory / name for name in ("item.json", "product.xml", "receipt.json")]
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        stac_bytes, xml_bytes = stac_path.read_bytes(), xml_path.read_bytes()
        if receipt["sentinel_id"] != identifier or digest(stac_bytes) != receipt["stac_sha256"] or digest(xml_bytes) != receipt["xml_sha256"]:
            raise ValueError("Cached metadata identity/hash mismatch; preserve and investigate cache")
        return json.loads(stac_bytes), xml_bytes, receipt
    if offline:
        raise FileNotFoundError(f"No complete cached metadata for {identifier}")
    item_url = f"{STAC}/collections/sentinel-2-l2a/items/{quote(identifier, safe='')}"
    stac_bytes = fetch(item_url)
    item = json.loads(stac_bytes)
    if item.get("id") != identifier:
        raise ValueError("Unexpected STAC response identifier")
    href = unsigned(item["assets"]["product-metadata"]["href"])
    parsed_url = urlsplit(href)
    if parsed_url.scheme != "https" or not (parsed_url.hostname or "").endswith(".blob.core.windows.net"):
        raise ValueError("Unexpected product metadata host/protocol")
    xml_bytes = fetch(signed_metadata_url(href))
    # Preserve the evidence even if its contents fail the subsequent assessment.
    # Signed URLs are never persisted.
    receipt = {"sentinel_id": identifier, "stac_url": item_url, "xml_url": href,
               "retrieved_utc": datetime.now(timezone.utc).isoformat(),
               "stac_sha256": digest(stac_bytes), "xml_sha256": digest(xml_bytes)}
    atomic_write(stac_path, stac_bytes)
    atomic_write(xml_path, xml_bytes)
    write_json(receipt_path, receipt)
    return item, xml_bytes, receipt


def run_audit(manifest_path: Path, output: Path, *, workers: int = 4,
              limit: int | None = None, offline: bool = False) -> dict:
    manifest_bytes = manifest_path.read_bytes()
    if digest(manifest_bytes) != EXPECTED_MANIFEST:
        raise ValueError("This tool requires the frozen Mahimandal release manifest")
    with manifest_path.open(encoding="utf-8-sig", newline="") as stream:
        manifest = list(csv.DictReader(stream))
    if len(manifest) != 4800 or len({row["pair_id"] for row in manifest}) != 4800:
        raise ValueError("Unexpected frozen pair membership")
    products = Counter(row["sentinel_id"] for row in manifest)
    identifiers = sorted(products)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        identifiers = identifiers[:limit]
    output = output.resolve()
    # Never put an audit cache into the read-only release or its parent tree.
    if output.is_relative_to(manifest_path.resolve().parent) or manifest_path.resolve().is_relative_to(output):
        raise ValueError("Audit output must be separate from the frozen release")
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "source_metadata"

    def check(identifier):
        try:
            item, xml, receipt = cached_source(identifier, cache, offline)
            return {**assess(item, xml, identifier), "pair_count": products[identifier],
                    "xml_sha256": receipt["xml_sha256"], "stac_sha256": receipt["stac_sha256"],
                    "xml_url": receipt["xml_url"], "cache_key": digest(identifier.encode())[:24]}
        except Exception as exc:
            return {"sentinel_id": identifier, "pair_count": products[identifier],
                    "status": "unresolved", "error_type": type(exc).__name__, "error": str(exc)}

    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(check, identifiers):
            results.append(result)
            if len(results) % 25 == 0 or len(results) == len(identifiers):
                print(f"Products checked: {len(results)}/{len(identifiers)}; "
                      f"status: {dict(Counter(row['status'] for row in results))}", flush=True)
    write_json(output / "product_results.json", results)
    fields = ["sentinel_id", "pair_count", "status", "product_uri", "processing_baseline",
              "boa_quantification_value", "notebook_additive_offset_dn",
              "max_unclipped_formula_difference_uint16", *[band + "_boa_add_offset" for band in BANDS],
              "xml_sha256", "stac_sha256", "error_type", "error"]
    with (output / "product_results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            writer.writerow({**result, **{band + "_boa_add_offset": result.get("boa_add_offsets", {}).get(band, "") for band in BANDS}})
    counts = Counter(row["status"] for row in results)
    complete = len(results) == len(products) and counts["unresolved"] == 0
    summary = {
        "manifest_sha256": EXPECTED_MANIFEST, "total_pairs": 4800,
        "total_unique_sentinel_products": len(products), "checked_products": len(results),
        "represented_pairs": sum(row["pair_count"] for row in results),
        "complete_product_coverage": complete, "status_counts": dict(counts),
        "all_product_metadata_supports_conversion": complete and counts["conversion_disagrees_with_metadata"] == 0,
        "baseline_counts": dict(Counter(row.get("processing_baseline", "unresolved") for row in results)),
        "references": REFERENCES,
        "interpretation": "Checks independently recorded original-product XML coefficients against the notebook formula. Does not establish all-pixel correctness, asset harmonization, cloud masking, registration accuracy, or independent geodetic truth.",
        "script_sha256": digest(Path(__file__).read_bytes()),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "summary.json", summary)
    failures = [row for row in results if row["status"] != "metadata_supports_conversion"]
    write_json(output / "unresolved_or_disagreeing.json", failures)
    lines = [f"{digest(path.read_bytes())}  {path.relative_to(output).as_posix()}"
             for path in sorted(output.rglob("*")) if path.is_file() and path.name not in ("checksums.sha256",) and not path.name.endswith(".tmp")]
    atomic_write(output / "checksums.sha256", ("\n".join(lines) + "\n").encode("utf-8"))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, help="Smoke test only; never marks partial coverage complete")
    parser.add_argument("--offline", action="store_true", help="Revalidate cached XML/STAC; no network")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    summary = run_audit(args.manifest, args.output, workers=args.workers, limit=args.limit, offline=args.offline)
    print(json.dumps(summary, indent=2))
    if not summary["all_product_metadata_supports_conversion"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
