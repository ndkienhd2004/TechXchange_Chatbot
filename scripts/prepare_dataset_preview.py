from __future__ import annotations

"""Build normalized staging files from external datasets before DB import."""

import argparse
import csv
import json
import re
from collections import Counter
from collections import defaultdict
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

# Default dataset and output paths used by local preview script.
DEFAULT_ECOM_DIR = Path("/Users/kien/Downloads/dataset/E-Commerce Tech")
DEFAULT_PC_DIR = Path("/Users/kien/Downloads/dataset/TechXchange-Pc part data/dataset")
DEFAULT_OUTPUT_DIR = Path("/Users/kien/Codes/TechXchange_Chatbot/docs/import_preview")

# File filters and sample size for preview artifacts.
CSV_GLOB = "*.csv"
JSON_GLOB = "*.json"
SAMPLE_SIZE = 200


def normalize_space(value: Any) -> str:
    """Collapse extra whitespace and coerce value to trimmed string."""

    return " ".join(str(value or "").strip().split())


def slugify(value: str) -> str:
    """Convert text to URL/file-safe slug format."""

    lowered = normalize_space(value).lower()
    cleaned = re.sub(r"[^a-z0-9]+", "-", lowered)
    return cleaned.strip("-")


def humanize_slug(value: str) -> str:
    """Convert slug-like text to title-cased display text."""

    return normalize_space(value.replace("-", " ").replace("_", " ")).title()


def parse_price(value: Any) -> float | None:
    """Parse flexible price strings with commas/dots into float."""

    text = normalize_space(value)
    if not text:
        return None
    matches = re.findall(r"-?\d[\d,\.]*", text)
    if not matches:
        return None
    token = max(matches, key=len)
    if not token:
        return None
    if "," in token and "." in token:
        token = token.replace(",", "")
    elif "," in token and "." not in token:
        parts = token.split(",")
        if len(parts[-1]) == 2:
            token = "".join(parts[:-1]) + "." + parts[-1]
        else:
            token = "".join(parts)
    try:
        return round(float(token), 2)
    except ValueError:
        return None


def parse_float(value: Any) -> float | None:
    """Parse numeric float value from noisy text input."""

    text = normalize_space(value)
    if not text:
        return None
    text = re.sub(r"[^0-9.\-]", "", text)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_int(value: Any) -> int | None:
    """Parse integer value from noisy text input."""

    text = normalize_space(value)
    if not text:
        return None
    match = re.search(r"-?\d+", text.replace(",", ""))
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def infer_brand(name: str, fallback: str = "") -> str:
    """Infer brand from product name/category hint with small special-cases."""

    if fallback:
        return normalize_space(fallback).title()
    normalized = normalize_space(name)
    if not normalized:
        return "Unknown"
    first_two = normalized.split()[:2]
    joined = " ".join(first_two).lower()
    # Handle a few common two-token brands.
    if joined.startswith("be quiet"):
        return "be quiet!"
    if joined.startswith("western digital"):
        return "Western Digital"
    if joined.startswith("cooler master"):
        return "Cooler Master"
    token = normalized.split()[0]
    token = re.sub(r"[^A-Za-z0-9+.!-]", "", token)
    return token if token else "Unknown"


def split_category_and_brand(raw_category: str) -> tuple[str, str]:
    """Split raw category field into normalized category and optional brand hint."""

    parts = [normalize_space(part) for part in str(raw_category or "").split(",")]
    parts = [part for part in parts if part]
    if not parts:
        return ("Uncategorized", "")
    category = humanize_slug(parts[0])
    brand_hint = parts[1] if len(parts) > 1 else ""
    return (category, brand_hint)


def cleaned_specs(payload: dict[str, Any], excluded: set[str]) -> dict[str, Any]:
    """Drop empty/irrelevant keys from raw specs payload."""

    out: dict[str, Any] = {}
    for key, value in payload.items():
        if key in excluded:
            continue
        if value is None:
            continue
        if isinstance(value, str) and normalize_space(value).lower() in {"", "n/a", "na", "none", "null"}:
            continue
        out[str(key)] = value
    return out


def compact_specs_text(specs: dict[str, Any], limit: int = 10) -> str:
    """Build short semicolon-separated specs summary for content fields."""

    if not specs:
        return ""
    parts = []
    for index, (key, value) in enumerate(specs.items(), start=1):
        if index > limit:
            break
        parts.append(f"{key}: {value}")
    return "; ".join(parts)


def build_content(record: dict[str, Any]) -> str:
    """Render one normalized record into chat-ingestion document content."""

    lines = [
        f"Product name: {record['name']}",
        f"Category: {record['category_name']}",
        f"Brand: {record['brand_name']}",
        f"Price: {record['price'] if record['price'] is not None else 'N/A'} {record['currency']}",
        f"Rating: {record['rating'] if record['rating'] is not None else 'N/A'}",
        f"Review count: {record['review_count'] if record['review_count'] is not None else 0}",
    ]
    if record.get("description"):
        lines.append(f"Description: {record['description']}")
    if record.get("specs"):
        lines.append(f"Specs: {compact_specs_text(record['specs'])}")
    if record.get("product_link"):
        lines.append(f"Source URL: {record['product_link']}")
    return "\n".join(lines)


def load_ecommerce_csv_records(csv_paths: list[Path]) -> list[dict[str, Any]]:
    """Load and normalize records from ecommerce CSV sources."""

    rows: list[dict[str, Any]] = []
    for path in csv_paths:
        dataset_slug = slugify(path.stem) or "ecommerce"
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            for index, row in enumerate(reader, start=1):
                name = normalize_space(row.get("product_name"))
                if not name:
                    continue
                category_name, brand_hint = split_category_and_brand(
                    normalize_space(row.get("product_category"))
                )
                brand_name = infer_brand(name, brand_hint)
                price = parse_price(row.get("product_price"))
                rating = parse_float(row.get("product_ratings"))
                review_count = parse_int(row.get("rating_count"))
                description = normalize_space(row.get("description"))
                store_name = normalize_space(row.get("product_store"))
                specs = {
                    "product_store": store_name,
                    "product_category_raw": normalize_space(row.get("product_category")),
                    "date_raw": normalize_space(row.get("date")),
                }
                rows.append(
                    {
                        "external_id": f"csv:{dataset_slug}:{index}",
                        "source_dataset": f"csv::{path.name}",
                        "name": name,
                        "description": description,
                        "price": price,
                        "currency": "PKR",
                        "rating": rating,
                        "review_count": review_count,
                        "buyturn": review_count or 0,
                        "brand_name": brand_name,
                        "category_name": category_name,
                        "image_url": normalize_space(row.get("product_image")),
                        "product_link": normalize_space(row.get("product_link")),
                        "store_name": store_name,
                        "specs": specs,
                    }
                )
    return rows


def load_pc_json_records(json_paths: list[Path]) -> list[dict[str, Any]]:
    """Load and normalize records from PC-part JSON sources."""

    rows: list[dict[str, Any]] = []
    for path in json_paths:
        category_name = humanize_slug(path.stem)
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            continue
        dataset_slug = slugify(path.stem) or "pc-part"
        for index, item in enumerate(data, start=1):
            if not isinstance(item, dict):
                continue
            name = normalize_space(item.get("name"))
            if not name:
                continue
            price = parse_price(item.get("price"))
            specs = cleaned_specs(item, excluded={"name", "price"})
            rows.append(
                {
                    "external_id": f"pc:{dataset_slug}:{index}",
                    "source_dataset": f"pc_json::{path.name}",
                    "name": name,
                    "description": "",
                    "price": price,
                    "currency": "USD",
                    "rating": None,
                    "review_count": 0,
                    "buyturn": 0,
                    "brand_name": infer_brand(name),
                    "category_name": category_name,
                    "image_url": "",
                    "product_link": "",
                    "store_name": "",
                    "specs": specs,
                }
            )
    return rows


def build_catalogs(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Group product variants into deduplicated catalog candidates."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = f"{slugify(record['name'])}|{slugify(record['brand_name'])}|{slugify(record['category_name'])}"
        grouped[key].append(record)

    catalogs: list[dict[str, Any]] = []
    key_to_catalog: dict[str, str] = {}
    for index, (group_key, items) in enumerate(grouped.items(), start=1):
        first = items[0]
        prices = [item["price"] for item in items if item.get("price") is not None]
        catalog_key = f"catalog:{index}"
        key_to_catalog[group_key] = catalog_key
        catalogs.append(
            {
                "catalog_key": catalog_key,
                "name": first["name"],
                "brand_name": first["brand_name"],
                "category_name": first["category_name"],
                "description": first.get("description") or "",
                "default_image": first.get("image_url") or "",
                "msrp_min": min(prices) if prices else None,
                "msrp_max": max(prices) if prices else None,
                "product_count": len(items),
                "sources": sorted({item["source_dataset"] for item in items}),
            }
        )
    return catalogs, key_to_catalog


def build_output_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compose all staging outputs (products/brands/categories/catalogs/docs)."""

    catalogs, key_to_catalog = build_catalogs(records)

    products: list[dict[str, Any]] = []
    for record in records:
        group_key = f"{slugify(record['name'])}|{slugify(record['brand_name'])}|{slugify(record['category_name'])}"
        products.append(
            {
                "external_id": record["external_id"],
                "source_dataset": record["source_dataset"],
                "name": record["name"],
                "description": record.get("description") or "",
                "price": record.get("price"),
                "currency": record["currency"],
                "brand_name": record["brand_name"],
                "category_name": record["category_name"],
                "catalog_key": key_to_catalog[group_key],
                "quality": "new",
                "condition_percent": 100,
                "rating": record.get("rating"),
                "buyturn": int(record.get("buyturn") or 0),
                "quantity": 20,
                "status": "active",
                "image_url": record.get("image_url") or "",
                "product_link": record.get("product_link") or "",
                "store_name": record.get("store_name") or "",
                "specs": record.get("specs") or {},
            }
        )

    brands = [
        {
            "name": brand_name,
            "slug": slugify(brand_name),
            "product_count": count,
        }
        for brand_name, count in sorted(
            Counter(item["brand_name"] for item in products).items(),
            key=lambda pair: pair[1],
            reverse=True,
        )
    ]

    categories = [
        {
            "name": category_name,
            "slug": slugify(category_name),
            "product_count": count,
        }
        for category_name, count in sorted(
            Counter(item["category_name"] for item in products).items(),
            key=lambda pair: pair[1],
            reverse=True,
        )
    ]

    now_iso = datetime.now(timezone.utc).date().isoformat()
    chatbot_docs = [
        {
            "source_key": f"dataset:{item['external_id']}",
            "title": item["name"],
            "uri": item["product_link"] or f"/dataset/{slugify(item['external_id'])}",
            "content": build_content(
                {
                    **item,
                    "review_count": item.get("buyturn", 0),
                }
            ),
            "metadata": {
                "doc_type": "product",
                "category": item["category_name"],
                "brand": item["brand_name"],
                "trust_score": 0.82,
                "updated_at": now_iso,
                "source": "external_dataset",
                "external_id": item["external_id"],
            },
        }
        for item in products
    ]

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_products": len(products),
        "total_brands": len(brands),
        "total_categories": len(categories),
        "total_catalogs": len(catalogs),
        "top_sources": Counter(item["source_dataset"] for item in products).most_common(20),
        "top_categories": Counter(item["category_name"] for item in products).most_common(20),
        "top_brands": Counter(item["brand_name"] for item in products).most_common(20),
        "notes": [
            "This preview is staging data for review, not direct SQL insert scripts.",
            "products table in backend still requires seller_id and store_id mapping at import time.",
            "Brand inference from product name is heuristic for PC-part JSON files.",
        ],
    }

    return {
        "products": products,
        "brands": brands,
        "categories": categories,
        "catalogs": catalogs,
        "chatbot_docs": chatbot_docs,
        "summary": summary,
    }


def write_json(path: Path, payload: Any) -> None:
    """Write pretty UTF-8 JSON file."""

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write newline-delimited JSON rows."""

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_preview_readme(output_dir: Path, summary: dict[str, Any]) -> None:
    """Generate README describing preview files and import notes."""

    lines = [
        "# DATASET PREVIEW FOR DB IMPORT",
        "",
        "This folder is generated by `scripts/prepare_dataset_preview.py`.",
        "",
        "## Files",
        "- `summary.json`: high-level stats and assumptions.",
        "- `products_staging.jsonl`: full normalized product rows.",
        "- `products_staging_sample.json`: first 200 product rows for quick review.",
        "- `brands_staging.json`: unique brands with counts.",
        "- `categories_staging.json`: unique categories with counts.",
        "- `catalogs_staging.json`: deduplicated catalog candidates.",
        "- `chatbot_ingest_preview.json`: documents ready for `/api/assistant/ingest`.",
        "- `chatbot_ingest_sample.json`: first 200 docs for quick review.",
        "",
        "## Stats",
        f"- total_products: {summary['total_products']}",
        f"- total_brands: {summary['total_brands']}",
        f"- total_categories: {summary['total_categories']}",
        f"- total_catalogs: {summary['total_catalogs']}",
        "",
        "## Import notes",
        "- This output is review-first staging data.",
        "- If importing into backend `products`, you still need seller/store mapping.",
        "- `brand_name` in PC-part records is heuristic and should be reviewed.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def build_balanced_sample(products: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    """Build sample containing both CSV and PC-part records when possible."""

    csv_rows = [row for row in products if str(row.get("external_id", "")).startswith("csv:")]
    pc_rows = [row for row in products if str(row.get("external_id", "")).startswith("pc:")]
    if not csv_rows or not pc_rows:
        return products[:size]

    csv_quota = size // 2
    pc_quota = size - csv_quota
    sample = csv_rows[:csv_quota] + pc_rows[:pc_quota]
    return sample[:size]


def main() -> None:
    """CLI entrypoint that reads datasets and writes preview artifacts."""

    parser = argparse.ArgumentParser(description="Prepare DB preview files from external datasets.")
    parser.add_argument("--ecom-dir", type=Path, default=DEFAULT_ECOM_DIR)
    parser.add_argument("--pc-dir", type=Path, default=DEFAULT_PC_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    ecom_paths = sorted(args.ecom_dir.glob(CSV_GLOB))
    pc_paths = sorted(args.pc_dir.glob(JSON_GLOB))
    if not ecom_paths and not pc_paths:
        raise SystemExit("No dataset files found.")

    raw_records = []
    raw_records.extend(load_ecommerce_csv_records(ecom_paths))
    raw_records.extend(load_pc_json_records(pc_paths))
    outputs = build_output_records(raw_records)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "summary.json", outputs["summary"])
    write_jsonl(args.output_dir / "products_staging.jsonl", outputs["products"])
    product_sample = build_balanced_sample(outputs["products"], SAMPLE_SIZE)
    write_json(args.output_dir / "products_staging_sample.json", product_sample)
    write_json(args.output_dir / "brands_staging.json", outputs["brands"])
    write_json(args.output_dir / "categories_staging.json", outputs["categories"])
    write_json(args.output_dir / "catalogs_staging.json", outputs["catalogs"])
    write_json(args.output_dir / "chatbot_ingest_preview.json", outputs["chatbot_docs"])
    sample_keys = {item["external_id"] for item in product_sample}
    chatbot_sample = [
        item
        for item in outputs["chatbot_docs"]
        if str(item.get("metadata", {}).get("external_id")) in sample_keys
    ][:SAMPLE_SIZE]
    write_json(args.output_dir / "chatbot_ingest_sample.json", chatbot_sample)
    write_preview_readme(args.output_dir, outputs["summary"])

    print(f"Generated preview files in: {args.output_dir}")
    print(f"Total products: {outputs['summary']['total_products']}")
    print(f"Total chatbot docs: {len(outputs['chatbot_docs'])}")


if __name__ == "__main__":
    main()
