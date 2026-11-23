"""Build the two vector collections described in PLAN.md SS2/SS3:
discovery cards (cheap shortlist scan) and field chunks (detail retrieval).

Input:
  data/raw/{latest scrape date}/{slug}.json    (scrape_myscheme.py)
  data/processed/eligibility_structured.json    (extract_eligibility.py)
Output:
  data/vector_index/{discovery_cards,field_chunks}.{json,npy}
    (read by app/services/vector_store.py; embedded locally with the ONNX
    model in data/embedding_model/ - no API key or GPU needed)

Chunking unit = one scheme's one field (SS2), not a fixed-size window - the
myScheme payload already arrives split into fields (eligibility, benefits,
application steps, documents), so those boundaries ARE the chunk boundaries.
A secondary character-based split only kicks in for the rare unusually-long
field. Every chunk is prefixed with the scheme name + field label before
embedding, and carries the full structured eligibility record as metadata
so retrieval can filter without a join.

Note: Chroma metadata values must be str/int/float/bool - no lists, no
None. List-valued eligibility facets (caste, gender, ...) are joined with
"|" and parsed back out with split("|") wherever they're read. A missing
metadata key (rather than a null) means "no constraint from this field" -
consistent with the fail-open filter design in PLAN.md SS3.
"""

import json
import re
import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # backend/, for app.*
from app.services.vector_store import INDEX_DIR, LocalEmbedder, write_collection

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

LONG_FIELD_CHARS = 1400  # secondary-split threshold (SS2: ~300-400 tokens)
SPLIT_OVERLAP_CHARS = 200  # ~15%

FIELD_LABELS = {
    "description": "Description",
    "eligibility": "Eligibility",
    "benefits": "Benefits",
    "exclusions": "Exclusions",
    "application": "How to Apply",
    "documents": "Documents Required",
    "faq": "FAQ",
}

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=LONG_FIELD_CHARS, chunk_overlap=SPLIT_OVERLAP_CHARS
)

# myScheme sometimes stores a bare "<br>\n\n" (or, in Hindi, its transliterated
# placeholder "<बीआर>") in a field when there's no real content, instead of an
# empty string. Strip any HTML-tag-shaped text (handles both) and drop
# whatever's left if it's too short to be a real chunk, rather than indexing
# noise as if it were scheme content.
_HTML_TAG_RE = re.compile(r"<[^>]+>")
MIN_MEANINGFUL_CHARS = 8


def _clean(text):
    return _HTML_TAG_RE.sub(" ", text).strip()


def latest_scrape_dir():
    raw_dir = DATA_DIR / "raw"
    dated_dirs = sorted((d for d in raw_dir.iterdir() if d.is_dir()), reverse=True)
    if not dated_dirs:
        raise SystemExit(f"No scrape snapshots found under {raw_dir}")
    return dated_dirs[0]


def load_eligibility_by_slug():
    path = DATA_DIR / "processed" / "eligibility_structured.json"
    if not path.exists():
        raise SystemExit(f"{path} not found - run extract_eligibility.py first")
    records = json.loads(path.read_text(encoding="utf-8"))
    return {r["slug"]: r for r in records}


def _join(values):
    return "|".join(str(v) for v in values) if values else ""


def eligibility_metadata(elig):
    """Flatten one eligibility_structured.json record into Chroma-safe
    scalar metadata. Keys are omitted (not set to None) when unknown."""
    meta = {
        "caste": _join(elig.get("caste")),
        "gender": _join(elig.get("gender")),
        "marital_status": _join(elig.get("marital_status")),
        "residence": _join(elig.get("residence")),
        "occupation_target_group": _join(elig.get("occupation_target_group")),
        "benefit_type_fine": _join(elig.get("benefit_type_fine")),
        "benefit_type_coarse": elig.get("benefit_type_coarse") or "",
        "education_level": elig.get("education_level") or "",
    }
    for key in (
        "age_min",
        "age_max",
        "family_income_ceiling",
        "individual_income_ceiling",
        "parent_income_ceiling",
    ):
        value = elig.get(key)
        if value is not None:
            meta[key] = value
    return meta


def base_metadata(raw, elig):
    basic = raw.get("en", {}).get("basicDetails", {})
    basic_hi = raw.get("hi", {}).get("basicDetails", {})
    meta = {
        "scheme_id": raw["scheme_id"],
        "slug": raw["slug"],
        "scheme_name": basic.get("schemeName") or raw["slug"],
        # kept alongside the (always-English) scheme_name so fuzzy scheme-
        # NAME matching (app/services/rag_service.py) can match a Devanagari-script
        # query too, not just Hinglish/English ones.
        "scheme_name_hi": basic_hi.get("schemeName") or "",
        "category": _join(elig.get("category")),
        "department": elig.get("department") or "",
        "level": elig.get("level") or "",
    }
    meta.update(eligibility_metadata(elig))
    return meta


def discovery_card_text(raw, elig, lang):
    lang_block = raw.get(lang, {})
    basic = lang_block.get("basicDetails", {})
    content = lang_block.get("schemeContent", {})
    scheme_name = basic.get("schemeName") or raw["slug"]
    group = ", ".join(elig.get("occupation_target_group") or []) or "General public"
    benefit = ", ".join(elig.get("benefit_type_fine") or []) or (elig.get("benefit_type_coarse") or "")
    category = ", ".join(elig.get("category") or [])
    brief = content.get("briefDescription", "")
    return f"{scheme_name} — For: {group} — Gives: {benefit} — Category: {category}. {brief}".strip()


def _field_texts(raw, lang):
    """Yield (field_type, text) pairs for one scheme in one language."""
    lang_block = raw.get(lang, {})
    content = lang_block.get("schemeContent", {})
    elig_desc = lang_block.get("eligibilityCriteria", {})

    if content.get("detailedDescription_md"):
        yield "description", content["detailedDescription_md"]
    if elig_desc.get("eligibilityDescription_md"):
        yield "eligibility", elig_desc["eligibilityDescription_md"]
    if content.get("benefits_md"):
        yield "benefits", content["benefits_md"]
    if content.get("exclusions_md"):
        yield "exclusions", content["exclusions_md"]

    for step in lang_block.get("applicationProcess", []) or []:
        if step.get("process_md"):
            mode = step.get("mode", "")
            yield "application", f"({mode}) {step['process_md']}"

    docs_text = raw.get("documents", {}).get(lang, {}).get("documentsRequired_md")
    if docs_text:
        yield "documents", docs_text

    for faq in raw.get("faqs", {}).get(lang, {}).get("faqs", []) or []:
        q, a = faq.get("question", ""), faq.get("answer_md", "")
        if q or a:
            yield "faq", f"{q}\n{a}"


def build_field_chunks(raw, elig, lang):
    scheme_name = raw.get(lang, {}).get("basicDetails", {}).get("schemeName") or raw["slug"]
    meta = base_metadata(raw, elig)
    meta["lang"] = lang

    docs = []
    # a field_type can occur more than once per scheme (multiple FAQs, one
    # application chunk per mode) - track occurrences separately from the
    # secondary-split index so chunk_index stays unique per field_type
    # instead of every occurrence resetting to 0.
    occurrence_counters = {}
    for field_type, text in _field_texts(raw, lang):
        text = _clean(text)
        if len(text) < MIN_MEANINGFUL_CHARS:
            continue
        label = FIELD_LABELS.get(field_type, field_type)
        prefixed = f"{scheme_name} — {label}: {text}"

        occurrence = occurrence_counters.get(field_type, 0)
        occurrence_counters[field_type] = occurrence + 1

        pieces = (
            _splitter.split_text(prefixed) if len(prefixed) > LONG_FIELD_CHARS else [prefixed]
        )
        for i, piece in enumerate(pieces):
            chunk_index = f"{occurrence}-{i}"
            chunk_meta = dict(meta, field_type=field_type, chunk_index=chunk_index)
            docs.append(Document(page_content=piece, metadata=chunk_meta))
    return docs


def build_discovery_cards(raw, elig):
    docs = []
    for lang in ("en", "hi"):
        text = discovery_card_text(raw, elig, lang)
        meta = base_metadata(raw, elig)
        meta["lang"] = lang
        meta["field_type"] = "discovery_card"
        docs.append(Document(page_content=text, metadata=meta))
    return docs


def main():
    scrape_dir = latest_scrape_dir()
    eligibility_by_slug = load_eligibility_by_slug()

    discovery_docs = []
    field_docs = []
    for path in sorted(scrape_dir.glob("*.json")):
        if path.name == "_listing.json":
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        elig = eligibility_by_slug.get(raw["slug"])
        if elig is None:
            print(f"  skipping {raw['slug']}: no eligibility record (run extract_eligibility.py again)")
            continue

        discovery_docs.extend(build_discovery_cards(raw, elig))
        for lang in ("en", "hi"):
            field_docs.extend(build_field_chunks(raw, elig, lang))

    print(f"Discovery cards: {len(discovery_docs)}")
    print(f"Field chunks: {len(field_docs)}")

    print("Loading local embedding model (data/embedding_model)...")
    embedder = LocalEmbedder()

    print("Embedding + storing discovery cards...")
    write_collection("discovery_cards", discovery_docs, embedder)

    print("Embedding + storing field chunks...")
    write_collection("field_chunks", field_docs, embedder)

    print(f"Done. Knowledge base written to {INDEX_DIR}")


if __name__ == "__main__":
    main()
