"""Deterministic regression checks for the scrape -> extract -> chunk -> embed
pipeline. No LLM calls (see PLAN.md's token-budget note) - this catches
data-quality and pipeline-idempotency bugs by direct inspection, plus a
small retrieval smoke-test baseline against the current (Phase 1,
semantic-only) knowledge base.

Written after manually finding three real issues while stress-testing
query_kb.py (2026-08-05):
  1. Re-running build_knowledge_base.py duplicated every chunk instead of
     replacing them (from_documents() appends to an existing collection).
  2. myScheme stores a bare "<br>" (or its Hindi-transliterated placeholder
     "<बीआर>") in a field when there's no real content, instead of an empty
     string - these were leaking into the index as meaningless chunks.
  3. The "farmer" occupation keyword list matched the bare stem
     "agricultur", which over-matched schemes merely tagged with the
     "Agriculture" topic category (e.g. a home-herb-garden health scheme)
     as if they were farmer-targeted schemes.

Run: python eval/golden_set.py
Exits non-zero if any check fails.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA_DIR = ROOT / "data"
INDEX_DIR = DATA_DIR / "vector_index"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

HTML_TAG_RE = re.compile(r"<[^>]+>")
MIN_MEANINGFUL_CHARS = 8

failures = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        failures.append(name)


def latest_scrape_dir():
    raw_dir = DATA_DIR / "raw"
    dirs = sorted((d for d in raw_dir.iterdir() if d.is_dir()), reverse=True)
    return dirs[0] if dirs else None


def load_raw_records(scrape_dir):
    records = {}
    for path in scrape_dir.glob("*.json"):
        if path.name == "_listing.json":
            continue
        records[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return records


def load_eligibility():
    path = DATA_DIR / "processed" / "eligibility_structured.json"
    if not path.exists():
        return {}
    return {r["slug"]: r for r in json.loads(path.read_text(encoding="utf-8"))}


# ---------------------------------------------------------------------------
# Section 1: raw scrape <-> processed eligibility completeness
# ---------------------------------------------------------------------------

def check_corpus_completeness(raw_records, eligibility_by_slug):
    missing = [slug for slug in raw_records if slug not in eligibility_by_slug]
    check(
        "every scraped scheme has an eligibility_structured.json record",
        not missing,
        f"missing: {missing[:10]}" if missing else "",
    )


# ---------------------------------------------------------------------------
# Section 2: known-case eligibility/occupation tagging assertions
# ---------------------------------------------------------------------------

KNOWN_ELIGIBILITY_CASES = [
    # (slug, field, expected_check)
    ("cmenspy", "gender", lambda v: "female" in v),
    ("cmenspy", "marital_status", lambda v: "widowed" in v),
    ("cmenspy", "benefit_type_fine", lambda v: "pension" in v),
    ("dpmsc", "caste", lambda v: v == ["sc"]),
    ("rcmrf-mh", "occupation_target_group", lambda v: v == []),
    # regression test for the "agricultur" stem over-match bug:
    ("ggay", "occupation_target_group", lambda v: "farmer" not in v),
    # positive control - the fix must not have removed real farmer matches:
    ("ky", "occupation_target_group", lambda v: "farmer" in v),
]


def check_known_eligibility_cases(eligibility_by_slug):
    for slug, field, predicate in KNOWN_ELIGIBILITY_CASES:
        record = eligibility_by_slug.get(slug)
        if record is None:
            check(f"{slug}.{field} known-case", False, "scheme not found in eligibility records")
            continue
        value = record.get(field)
        check(f"{slug}.{field} known-case", predicate(value), f"got {value!r}")


# ---------------------------------------------------------------------------
# Section 3: knowledge-base-level checks (require the built Chroma DB)
# ---------------------------------------------------------------------------

def _load_chroma_collections():
    # Name kept for history; now the local vector index (app/services/vector_store.py).
    sys.path.insert(0, str(ROOT))
    from app.services.vector_store import LocalEmbedder, VectorCollection

    embedder = LocalEmbedder()
    return VectorCollection("discovery_cards", embedder), VectorCollection("field_chunks", embedder)


def check_no_duplicate_chunks(chunks):
    all_docs = chunks.get(include=["metadatas"])
    keys = [
        (m["scheme_id"], m["field_type"], m["lang"], m.get("chunk_index", 0))
        for m in all_docs["metadatas"]
    ]
    counts = Counter(keys)
    dupes = [k for k, c in counts.items() if c > 1]
    check(
        "no duplicate chunks in field_chunks (rebuild-idempotency regression)",
        not dupes,
        f"{len(dupes)} duplicate keys, e.g. {dupes[:3]}" if dupes else "",
    )


def check_no_junk_chunks(cards, chunks):
    bad = []
    for collection, label in ((cards, "discovery_cards"), (chunks, "field_chunks")):
        all_docs = collection.get(include=["metadatas", "documents"])
        for doc_text, meta in zip(all_docs["documents"], all_docs["metadatas"]):
            prefix = f"{meta['scheme_name']} — "
            body = doc_text[len(prefix):] if doc_text.startswith(prefix) else doc_text
            cleaned = HTML_TAG_RE.sub(" ", body).strip()
            # strip a leading "Label: " if present, to check the actual content length
            cleaned = re.sub(r"^[^:]{1,30}:\s*", "", cleaned)
            if len(cleaned) < MIN_MEANINGFUL_CHARS:
                bad.append(f"{label}/{meta['scheme_name']}/{meta.get('field_type')}/{meta['lang']}")
    check(
        "no near-empty/HTML-placeholder-only chunks in the index",
        not bad,
        f"{len(bad)} junk chunks, e.g. {bad[:5]}" if bad else "",
    )


def check_corpus_size(cards, chunks):
    n_cards = cards.count()
    n_chunks = chunks.count()
    check("discovery_cards collection is non-trivially sized", n_cards > 100, f"count={n_cards}")
    check("field_chunks collection is non-trivially sized", n_chunks > 1000, f"count={n_chunks}")


# ---------------------------------------------------------------------------
# Section 4: retrieval smoke-test baseline (semantic-only, Phase 1 state)
# ---------------------------------------------------------------------------

# Loose bounds, not "must be #1" - Phase 1 has no eligibility filter/rank
# boost yet (that's Phase 2). These lock in what's ALREADY good, and track
# (without falsely asserting fixed) the known-hard cases that motivate
# Phase 2, so a future change can't silently make them worse.
RETRIEVAL_SMOKE_CASES = [
    ("SC scholarship for students", "Ambedkar International Scholarship", 3),
    ("krishi yantra subsidy for farmer", "Krishi Yantra", 3),
    ("pension for widow", "Chief Minister Ekal Nari Samman Pension Yojana", 20),  # known Phase-2-needed case
]


def check_retrieval_smoke(cards):
    for query, expected_substring, max_rank in RETRIEVAL_SMOKE_CASES:
        results = cards.similarity_search(query, k=max_rank)
        found = any(expected_substring in doc.metadata["scheme_name"] for doc in results)
        check(
            f"retrieval smoke: '{query}' surfaces '{expected_substring}' in top {max_rank}",
            found,
        )


# ---------------------------------------------------------------------------
# Section 5b: category coverage - every real scheme category (not just the
# handful of cases stumbled onto manually) gets a sanity check: (a) at least
# one scraped scheme actually carries that category, and (b) a natural
# query for that category surfaces a same-category scheme near the top of
# Stage A's shortlist. Catches "category X quietly stopped retrieving
# anything sensible" the same way Section 4 catches it for individual
# hand-picked queries, but systematically across the whole category list
# instead of whichever ones happened to get manually tested.
#
# The category strings are copied verbatim from eligibility_structured.json
# (2026-08-05 scrape) rather than retyped, since exact string matching is
# how rag_service's category-boost works - a retyped near-miss (extra
# space, different ampersand spacing) would silently fail to match.
# ---------------------------------------------------------------------------

CATEGORY_QUERIES = {
    "Education & Learning": [
        "scholarship for students",
        "coaching for competitive exam",
        "hostel for tribal students",
    ],
    "Social welfare & Empowerment": [
        "financial help for poor family",
        "pension for senior citizen",
        "widow support scheme",
    ],
    "Agriculture,Rural & Environment": [
        "farmer subsidy",
        "irrigation pipeline scheme",
        "greenhouse farming assistance",
    ],
    "Women and Child": [
        "scheme for women and children",
        "maternity benefit scheme",
        "girl child scooty scheme",
    ],
    "Skills & Employment": [
        "job training scheme",
        "self employment training",
        "skill development for youth",
    ],
    "Health & Wellness": [
        "medical treatment assistance",
        "accident relief fund",
        "silicosis patient support",
    ],
    "Business & Entrepreneurship": [
        "loan for starting a business",
        "technology upgrade assistance for business",
        "interest subsidy for entrepreneurs",
    ],
    "Housing & Shelter": [
        "house construction scheme",
        "hostel rent assistance",
        "shelter for tribal students",
    ],
    "Banking,Financial Services and Insurance": [
        "insurance scheme",
        "old age pension",
        "widow pension",
    ],
    "Sports & Culture": [
        "scheme for sportsperson",
        "sports hostel for tribal students",
        "encouragement for tribal sports players",
    ],
    "Travel & Tourism": [
        "pilgrimage yatra assistance",
        "kailash mansarovar yatra scheme",
        "religious tourism assistance",
    ],
}

# Per-category eligibility-filter contrast cases, each grounded in a real,
# verified restrictive scheme found in eligibility_structured.json
# (2026-08-05 scrape) - not invented/assumed. citizen_kwargs describes a
# citizen who should NOT qualify for excluded_scheme_substring; each case
# also names a same-category scheme that SHOULD survive, to confirm the
# filter isn't over-excluding the whole category.
CATEGORY_ELIGIBILITY_CASES = [
    ("Education & Learning", {"caste": "obc"}, "Ambedkar Fellowship Scheme", "Devnarayan Pratibhawan Chhatra Protsahan Yojana"),
    ("Social welfare & Empowerment", {"gender": "male"}, "Transgender Utthan Kosh", "World War II Pension"),
    ("Agriculture,Rural & Environment", {"gender": "male"}, "Incentive To Girls", "Krishi Yantra"),
    ("Women and Child", {"gender": "male"}, "Mukhyamantri Rajshri Yojana", None),
    ("Skills & Employment", {"gender": "male"}, "Shubhshakti Yojana", "Motorized Tricycle Yojana"),
    ("Health & Wellness", {"age": 15}, "Silicosis Pidit", "Rajasthan Chief Minister's Relief Fund"),
    # note: virtually every scheme in this category requires age>=18, so age
    # isn't a useful "must survive" differentiator here - use gender instead.
    ("Business & Entrepreneurship", {"gender": "male"}, "Transgender Utthan Kosh - Self Employment", "Technology Acquisition Assistance"),
    ("Housing & Shelter", {"caste": "general"}, "Hostel Rent Assistance Scheme for Tribal Students", "Gaushala Vikas Yojana"),
    ("Banking,Financial Services and Insurance", {"gender": "male"}, "Indira Gandhi National Vidhva Pension", "Jan Samuh Bima Yojana"),
    ("Sports & Culture", {"caste": "general"}, "Cash Amount for Encouraging Scheduled Tribe Sports", "Nirman Shramik Antrashtriya Khel"),
    ("Travel & Tourism", {"age": 15}, "Assistance to Pilgrims for Kailash Mansarovar", None),
]


def check_category_coverage(eligibility_by_slug, cards):
    from app.services.rag_service import CitizenContext

    present_categories = set()
    for record in eligibility_by_slug.values():
        present_categories.update(record.get("category") or [])

    for category in CATEGORY_QUERIES:
        check(
            f"corpus has >=1 scheme tagged '{category}'",
            category in present_categories,
        )

    for category, queries in CATEGORY_QUERIES.items():
        for query in queries:
            citizen = CitizenContext(need_category=category)
            results = _shortlist_for_category_check(cards, citizen, query, k=5)
            found = any(category in (r.metadata.get("category") or "") for r in results)
            check(
                f"category boost: '{query}' surfaces a '{category}' scheme in top 5",
                found,
                "" if found else f"top 5: {[r.scheme_name for r in results]}",
            )


def check_category_eligibility_filter(cards):
    """Tests eligibility_verdict() directly against each named scheme's
    real metadata (looked up by name, not via semantic search) - decoupled
    from retrieval quality on purpose, so this only fails when the FILTER
    logic itself is wrong, not when a query happens not to surface a
    scheme semantically (that's Section 5b's job, separately)."""
    from app.services.rag_service import CitizenContext, eligibility_verdict

    all_docs = cards.get(include=["metadatas"])
    meta_by_scheme_name = {}
    for meta in all_docs["metadatas"]:
        meta_by_scheme_name.setdefault(meta["scheme_name"], meta)

    def _find(substring):
        for name, meta in meta_by_scheme_name.items():
            if substring in name:
                return name, meta
        return None, None

    for category, citizen_kwargs, excluded_substring, must_survive_substring in CATEGORY_ELIGIBILITY_CASES:
        citizen = CitizenContext(**citizen_kwargs)

        excluded_name, excluded_meta = _find(excluded_substring)
        if excluded_meta is None:
            check(f"{category}: '{excluded_substring}' exists in the corpus", False)
            continue
        eligible, reason = eligibility_verdict(excluded_meta, citizen)
        check(
            f"{category}: {citizen_kwargs} correctly excludes '{excluded_name}'",
            not eligible,
            "" if not eligible else "was NOT excluded - filter let it through",
        )

        if must_survive_substring:
            survive_name, survive_meta = _find(must_survive_substring)
            if survive_meta is None:
                check(f"{category}: '{must_survive_substring}' exists in the corpus", False)
                continue
            eligible, reason = eligibility_verdict(survive_meta, citizen)
            check(
                f"{category}: {citizen_kwargs} doesn't over-exclude '{survive_name}'",
                eligible,
                "" if eligible else f"wrongly excluded: {reason}",
            )


def _shortlist_for_category_check(cards, citizen, query, k):
    """Mirrors RagService.shortlist's filter+boost+dedupe, using only the
    discovery_cards handle already loaded for Section 3/4, so this section
    doesn't need its own RagService (embedding model) instance."""
    from app.services.rag_service import eligibility_verdict, _rank_boost, ScoredScheme

    raw_hits = cards.similarity_search_with_score(query, k=SHORTLIST_POOL_SIZE)
    best_by_scheme = {}
    for doc, distance in raw_hits:
        meta = doc.metadata
        eligible, _ = eligibility_verdict(meta, citizen)
        if not eligible:
            continue
        adjusted = distance - _rank_boost(meta, citizen)
        existing = best_by_scheme.get(meta["scheme_id"])
        if existing is None or adjusted < existing.score:
            best_by_scheme[meta["scheme_id"]] = ScoredScheme(
                scheme_id=meta["scheme_id"],
                scheme_name=meta["scheme_name"],
                lang=meta["lang"],
                score=adjusted,
                metadata=meta,
            )
    return sorted(best_by_scheme.values(), key=lambda s: s.score)[:k]


SHORTLIST_POOL_SIZE = 100


# ---------------------------------------------------------------------------
# Section 6: hybrid name-match recall (app/services/rag_service.py) - regression for
# the bare-scheme-name gap found during Phase 3 conversation stress testing
# (2026-08-05): "krishi yantra" (the exact scheme's own name) ranked 9th of
# 76 eligible candidates via semantic search alone. A fuzzy name-match pass
# was added to pull/boost a strongly name-matched scheme regardless of its
# raw semantic rank - this locks in that it actually reaches top-1 now.
# ---------------------------------------------------------------------------

def check_hybrid_name_match():
    from app.services.rag_service import RagService, CitizenContext

    svc = RagService()
    citizen = CitizenContext()
    for query in ("krishi yantra", "krishi yantra ke baare mein batao", "tell me about krishi yantra"):
        results = svc.shortlist(query, citizen, top_n=1)
        top_name = results[0].scheme_name if results else None
        check(
            f"hybrid name-match: '{query}' ranks 'Krishi Yantra' #1",
            top_name == "Krishi Yantra",
            f"got {top_name!r}",
        )


# ---------------------------------------------------------------------------
# Section 7: education_level filter - regression for the live-found bug
# (2026-08-05) where an MCA graduate asking about higher-studies schemes got
# Class-6-8 and Class-10/12 scholarships mixed into the results. Fixed via a
# one-time LLM classification per education-relevant scheme (not a keyword
# table - most schemes, e.g. "Rajya Pratibha Khoj", don't name their level
# anywhere except free eligibility text). This checks the classification
# actually landed in the KB and that the filter uses it correctly.
# ---------------------------------------------------------------------------

EDUCATION_LEVEL_CASES = [
    # (scheme_name, expected_education_level)
    ("Rajya Pratibha Khoj Pariksha Chhatravriti Yojana", "school"),
    ("Schedule Caste Pre-Metric Scholarship Class 06 To 08", "school"),
    ("Mukhya Mantri Sarvajan Uchcha Shiksha Chhatravriti Yojana", "higher_education"),
]


def check_education_level_filter(cards):
    from app.services.rag_service import CitizenContext, eligibility_verdict

    meta_by_name = {}
    for meta in cards.get(include=["metadatas"])["metadatas"]:
        meta_by_name.setdefault(meta["scheme_name"], meta)

    for scheme_name, expected_level in EDUCATION_LEVEL_CASES:
        meta = meta_by_name.get(scheme_name)
        if meta is None:
            check(f"'{scheme_name}' exists in the corpus", False)
            continue
        check(
            f"'{scheme_name}' classified as education_level={expected_level!r}",
            meta.get("education_level") == expected_level,
            f"got {meta.get('education_level')!r}",
        )

    citizen = CitizenContext(education_level="higher_education")
    school_meta = meta_by_name.get("Rajya Pratibha Khoj Pariksha Chhatravriti Yojana")
    if school_meta:
        eligible, reason = eligibility_verdict(school_meta, citizen)
        check(
            "higher_education citizen correctly excluded from a school-level scheme",
            not eligible,
            "" if not eligible else "was NOT excluded",
        )

    higher_ed_meta = meta_by_name.get("Mukhya Mantri Sarvajan Uchcha Shiksha Chhatravriti Yojana")
    if higher_ed_meta:
        eligible, reason = eligibility_verdict(higher_ed_meta, citizen)
        check(
            "higher_education citizen NOT excluded from a higher-education scheme",
            eligible,
            "" if eligible else f"wrongly excluded: {reason}",
        )


# ---------------------------------------------------------------------------
# Section 5: Stage A eligibility filter (app/services/rag_service.py) - fail-open
# correctness, verified manually 2026-08-05 against real query results.
# ---------------------------------------------------------------------------

def check_eligibility_filter():
    from app.services.rag_service import RagService, CitizenContext

    svc = RagService()

    # A widowed woman must never be shown marriage-eligibility schemes that
    # require never-married/married status - this was verified manually
    # (12 correctly dropped) and is worth locking in as a regression.
    citizen = CitizenContext(gender="female", marital_status="widowed")
    trace = {}
    svc.shortlist("pension for widow", citizen, top_n=8, trace=trace)
    dropped_names = {d["scheme_name"] for d in trace["dropped"]}
    check(
        "widowed citizen: marriage-only schemes correctly excluded",
        any("Kanyadaan" in n or "Dampatya" in n for n in dropped_names),
        f"dropped: {dropped_names}",
    )

    # Fail-open: an UNKNOWN attribute must never exclude a scheme. A blank
    # CitizenContext should filter out nothing on eligibility grounds alone.
    blank_citizen = CitizenContext()
    trace2 = {}
    svc.shortlist("pension for widow", blank_citizen, top_n=8, trace=trace2)
    check(
        "fail-open: a citizen with no known facts has nothing excluded",
        len(trace2["dropped"]) == 0,
        f"{len(trace2['dropped'])} dropped with an empty profile",
    )


# ---------------------------------------------------------------------------
# Section 8: identity-aware search (app/services/rag_service.py) - regression for the
# 2026-08-10 fix where a citizen's already-known identity facts (e.g.
# marital_status=widowed) never reached the embedding query text - only the
# eligibility filter/rank boost saw them - so a widow asking generically for
# "financial assistance" never surfaced the real widow-pension schemes at
# all. Fixed by appending known identity terms to the Stage A search query.
# ---------------------------------------------------------------------------

IDENTITY_SEARCH_CASES = [
    # (citizen_kwargs, query, expected_substring, max_rank)
    ({"marital_status": "widowed", "gender": "female"}, "financial assistance", "Vidhva Pension", 10),
    ({"marital_status": "widowed", "gender": "female"}, "financial support", "Vidhva Pension", 10),
]


def check_identity_aware_search():
    from app.services.rag_service import RagService, CitizenContext

    svc = RagService()
    for citizen_kwargs, query, expected_substring, max_rank in IDENTITY_SEARCH_CASES:
        citizen = CitizenContext(**citizen_kwargs)
        results = svc.shortlist(query, citizen, top_n=max_rank)
        names = [r.scheme_name for r in results]
        found = any(expected_substring in n for n in names)
        check(
            f"identity-aware search: {citizen_kwargs} + '{query}' surfaces '{expected_substring}' in top {max_rank}",
            found,
            "" if found else f"top {max_rank}: {names}",
        )


# ---------------------------------------------------------------------------
# Section 9: generic-phrase name-match guard (app/services/rag_service.py) - regression
# for the 2026-08-10 fix where the hybrid name-match boost (built for bare
# scheme-name lookups like "krishi yantra") misfired on generic phrases that
# happen to be substrings of many scheme titles - "financial assistance"
# fuzzy-matched 6 different "Financial Assistance Scheme for X" titles at
# partial_ratio=100, crowding out unrelated-by-name but more relevant
# schemes. Fixed by only applying the boost when a query name-matches a
# small number of distinct schemes - a genuine name lookup names one thing.
# ---------------------------------------------------------------------------

def check_generic_phrase_name_guard():
    from app.services.rag_service import RagService, CitizenContext

    svc = RagService()
    citizen = CitizenContext()

    trace = {}
    svc.shortlist("financial assistance", citizen, top_n=5, trace=trace)
    check(
        "generic phrase 'financial assistance' does NOT trigger the name-match boost",
        trace["name_boost_applied"] is False,
        f"matched {len(trace['name_matches'])} distinct schemes: {trace['name_matches']}",
    )

    trace2 = {}
    svc.shortlist("krishi yantra", citizen, top_n=5, trace=trace2)
    check(
        "specific name lookup 'krishi yantra' STILL triggers the name-match boost",
        trace2["name_boost_applied"] is True,
        f"matched {len(trace2['name_matches'])} distinct schemes: {trace2['name_matches']}",
    )


def main():
    scrape_dir = latest_scrape_dir()
    if scrape_dir is None:
        sys.exit("No scrape snapshot found under data/raw/ - run scrape_myscheme.py first")

    print(f"Using scrape snapshot: {scrape_dir}\n")

    raw_records = load_raw_records(scrape_dir)
    eligibility_by_slug = load_eligibility()

    print("-- Section 1: corpus completeness --")
    check_corpus_completeness(raw_records, eligibility_by_slug)

    print("\n-- Section 2: known eligibility/occupation-tagging cases --")
    check_known_eligibility_cases(eligibility_by_slug)

    print("\n-- Section 3: knowledge-base checks (vector index) --")
    try:
        cards, chunks = _load_chroma_collections()
    except Exception as e:
        check("knowledge base is reachable", False, str(e))
        cards = chunks = None

    if cards is not None:
        check_corpus_size(cards, chunks)
        check_no_duplicate_chunks(chunks)
        check_no_junk_chunks(cards, chunks)

        print("\n-- Section 4: retrieval smoke-test baseline --")
        check_retrieval_smoke(cards)

        print("\n-- Section 5b: category coverage (all 11 categories, multiple queries each) --")
        check_category_coverage(eligibility_by_slug, cards)

        print("\n-- Section 5c: per-category eligibility filter contrast cases --")
        check_category_eligibility_filter(cards)

        print("\n-- Section 5: Stage A eligibility filter (rag_service.py) --")
        check_eligibility_filter()

        print("\n-- Section 6: hybrid name-match recall --")
        check_hybrid_name_match()

        print("\n-- Section 7: education_level filter --")
        check_education_level_filter(cards)

        print("\n-- Section 8: identity-aware search --")
        check_identity_aware_search()

        print("\n-- Section 9: generic-phrase name-match guard --")
        check_generic_phrase_name_guard()

    print(f"\n{len(failures)} failure(s).")
    if failures:
        for f in failures:
            print(" -", f)
        sys.exit(1)


if __name__ == "__main__":
    main()
