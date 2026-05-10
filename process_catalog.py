"""
process_catalog.py
------------------
Converts the raw SHL catalog JSON into a clean format our app can use.

Step 1 (run once on your machine):
  Download the catalog:
    curl "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json" -o catalog/raw_catalog.json

Step 2:
  python process_catalog.py

Output: catalog/shl_catalog.json  (this is what the rest of the app uses)
"""

import json
import os

RAW_FILE  = "catalog/raw_catalog.json"
OUT_FILE  = "catalog/shl_catalog.json"

# Map verbose key names → short single letters (matches assignment schema)
KEY_TO_TYPE = {
    "Ability & Aptitude":           "A",
    "Biodata & Situational Judgment": "B",
    "Competencies":                 "C",
    "Development & 360":            "D",
    "Assessment Exercises":         "E",
    "Knowledge & Skills":           "K",
    "Personality & Behavior":       "P",
    "Simulations":                  "S",
}


def parse_duration(raw: str) -> int | None:
    """Extract integer minutes from strings like 'Approximate Completion Time in minutes = 30'."""
    import re
    m = re.search(r"=\s*(?:max\s*)?(\d+)", raw or "")
    return int(m.group(1)) if m else None


def process(raw: list[dict]) -> list[dict]:
    clean = []
    for item in raw:
        if item.get("status") != "ok":
            continue  # skip any broken entries

        # Map list of verbose key names → list of single-letter codes
        type_codes = sorted(set(
            KEY_TO_TYPE[k] for k in item.get("keys", []) if k in KEY_TO_TYPE
        ))

        clean.append({
            "id":             item["entity_id"],
            "name":           item["name"],
            "url":            item["link"],          # always from scraped catalog = safe to return
            "description":    (item.get("description") or "").strip(),
            "test_type":      ", ".join(type_codes), # e.g. "K" or "K, S"
            "remote_testing": item.get("remote", "no").lower() == "yes",
            "adaptive":       item.get("adaptive", "no").lower() == "yes",
            "duration_minutes": parse_duration(item.get("duration_raw", "")),
            "job_levels":     item.get("job_levels") or [],
            "languages":      item.get("languages") or [],
        })

    return clean


def main():
    if not os.path.exists(RAW_FILE):
        print(f"ERROR: {RAW_FILE} not found.")
        print("Download it first with:")
        print(f'  curl "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json" -o {RAW_FILE}')
        return

    with open(RAW_FILE, encoding="utf-8") as f:
        raw = json.load(f, strict=False)

    print(f"Raw catalog:   {len(raw)} entries")

    clean = process(raw)
    print(f"Clean catalog: {len(clean)} entries (after filtering status != ok)")

    # Quick stats
    from collections import Counter
    type_counts = Counter(item["test_type"] for item in clean)
    print("\nTest type breakdown:")
    for t, count in sorted(type_counts.items()):
        print(f"  {t or '(none)'}: {count}")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)

    print(f"\nSaved → {OUT_FILE}")
    print("\nSample entry:")
    print(json.dumps(clean[0], indent=2))


if __name__ == "__main__":
    main()