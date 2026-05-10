"""
SHL Catalog Scraper
-------------------
Scrapes all Individual Test Solutions from:
  https://www.shl.com/solutions/products/product-catalog/

HOW TO RUN (on your local machine):
  pip install playwright beautifulsoup4
  python -m playwright install chromium
  python scraper.py

Output: catalog/shl_catalog.json
"""

import json
import time
import re
import os
# pyrefly: ignore [missing-import]
from playwright.sync_api import sync_playwright

# ── Config ───────────────────────────────────────────────────────────────────
BASE_URL    = "https://www.shl.com"
# type=1 filters to "Individual Test Solutions" only (not pre-packaged job solutions)
CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/?start={start}&type=1"
OUTPUT_FILE = "catalog/shl_catalog.json"
PAGE_SIZE   = 12    # SHL shows 12 items per page
MAX_PAGES   = 30    # safety cap


# ── Scrape one catalog listing page ──────────────────────────────────────────
def scrape_catalog_page(page, start: int) -> list[dict]:
    """
    Load one paginated catalog page.
    Returns list of basic assessment info (name, url, remote, adaptive, test_type).
    """
    url = CATALOG_URL.format(start=start)
    print(f"    Loading: {url}")
    page.goto(url, wait_until="networkidle", timeout=60000)

    # Wait for the table to appear (it's JS-rendered)
    try:
        page.wait_for_selector("table", timeout=15000)
    except Exception:
        print("    No table found on this page.")
        return []

    rows = []
    table_rows = page.query_selector_all("table tbody tr")

    # Fallback: try all tr if tbody doesn't exist
    if not table_rows:
        table_rows = page.query_selector_all("table tr")

    for row in table_rows:
        cells = row.query_selector_all("td")
        if len(cells) < 2:
            continue  # skip header or empty rows

        # Cell 0: Name + Link
        link_el = cells[0].query_selector("a")
        if not link_el:
            continue

        name = link_el.inner_text().strip()
        href = link_el.get_attribute("href") or ""
        full_url = href if href.startswith("http") else BASE_URL + href

        # Cell 1: Remote Testing (checkmark image = True)
        remote = len(cells) > 1 and bool(
            cells[1].query_selector("img, .fa-check, [class*='check'], span")
        )

        # Cell 2: Adaptive/IRT
        adaptive = len(cells) > 2 and bool(
            cells[2].query_selector("img, .fa-check, [class*='check'], span")
        )

        # Cell 3: Test Type letter (K, P, A, B, C, S...)
        test_type = cells[3].inner_text().strip() if len(cells) > 3 else ""

        rows.append({
            "name":           name,
            "url":            full_url,
            "remote_testing": remote,
            "adaptive":       adaptive,
            "test_type":      test_type,
        })

    return rows


# ── Scrape one product detail page ───────────────────────────────────────────
def scrape_product_detail(page, product_url: str) -> dict:
    """
    Visit a single assessment page.
    Returns: description, duration_minutes, job_levels, languages.
    """
    detail = {
        "description":      "",
        "duration_minutes": None,
        "job_levels":       [],
        "languages":        [],
    }

    try:
        page.goto(product_url, wait_until="networkidle", timeout=30000)
        time.sleep(0.8)  # let dynamic content settle

        # Description — try common selectors SHL uses
        for selector in [
            ".product-catalogue-training-calendar__row--description p",
            ".hero__description p",
            ".description p",
            "main article p",
            "main p",
        ]:
            el = page.query_selector(selector)
            if el:
                text = el.inner_text().strip()
                if len(text) > 30:
                    detail["description"] = text
                    break

        # Full page text for parsing other fields
        full_text = (
            page.inner_text("main")
            if page.query_selector("main")
            else page.inner_text("body")
        )

        # Duration — matches "36 minutes" or "Completion Time: 36 minutes"
        m = re.search(r"(\d+)\s*minutes?", full_text, re.IGNORECASE)
        if m:
            detail["duration_minutes"] = int(m.group(1))

        # Job Levels
        all_levels = [
            "Entry-Level", "Graduate", "Mid-Professional",
            "Professional Individual Contributor", "Manager",
            "Front Line Manager", "Supervisor", "Director", "Executive",
            "General Population",
        ]
        detail["job_levels"] = [
            lvl for lvl in all_levels if lvl.lower() in full_text.lower()
        ]

        # Languages
        all_langs = [
            "English", "Spanish", "French", "German", "Dutch", "Italian",
            "Portuguese", "Chinese", "Japanese", "Arabic", "Russian",
            "Korean", "Turkish", "Polish", "Swedish", "Danish", "Norwegian",
            "Finnish", "Czech", "Hungarian", "Romanian", "Bulgarian",
            "Croatian", "Serbian", "Slovak", "Thai", "Indonesian",
            "Malay", "Vietnamese", "Greek",
        ]
        detail["languages"] = [
            lang for lang in all_langs if lang.lower() in full_text.lower()
        ]

    except Exception as e:
        print(f"    Detail scrape failed for {product_url}: {e}")

    return detail


# ── Helpers ───────────────────────────────────────────────────────────────────
def save(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs("catalog", exist_ok=True)
    all_assessments = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            # Real browser user-agent so SHL doesn't block us
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )
        page = context.new_page()

        # Phase 1: Collect all names + URLs from the paginated listing
        print("=" * 55)
        print("Phase 1: Scraping catalog listing pages")
        print("=" * 55)

        for page_num in range(MAX_PAGES):
            start = page_num * PAGE_SIZE
            print(f"\n  Page {page_num + 1} (start={start})")

            rows = scrape_catalog_page(page, start)

            if not rows:
                print(f"  No more rows — stopped at page {page_num + 1}")
                break

            all_assessments.extend(rows)
            print(f"  Got {len(rows)} assessments (total: {len(all_assessments)})")
            time.sleep(1)  # be polite to the server

        print(f"\nTotal found: {len(all_assessments)} assessments")

        # Phase 2: Visit each product page for description + details
        print("\n" + "=" * 55)
        print("Phase 2: Scraping individual product pages")
        print("=" * 55)

        for i, assessment in enumerate(all_assessments):
            print(f"\n  [{i+1}/{len(all_assessments)}] {assessment['name']}")
            detail = scrape_product_detail(page, assessment["url"])
            assessment.update(detail)

            # Save every 10 items so progress isn't lost if it crashes
            if (i + 1) % 10 == 0:
                save(all_assessments, OUTPUT_FILE)
                print(f"  Progress saved ({i+1} done)")

            time.sleep(0.5)

        browser.close()

    # Final save
    save(all_assessments, OUTPUT_FILE)

    print("\n" + "=" * 55)
    print(f"Done! Saved {len(all_assessments)} assessments to {OUTPUT_FILE}")
    print("=" * 55)

    if all_assessments:
        print("\nSample entry:")
        print(json.dumps(all_assessments[0], indent=2))


if __name__ == "__main__":
    main()