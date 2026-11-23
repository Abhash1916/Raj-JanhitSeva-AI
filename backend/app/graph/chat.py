"""Phase 3 (PLAN.md SS4/SS11): the conversation graph.

Three layers of state, per SS4:
  a. profile            - persistent citizen facts, merged (never overwritten wholesale)
  b. need_text/category  - resets every turn, drives this turn's retrieval only
  c. shown_scheme_*      - short-lived, only for pronoun follow-ups

Life of one query (matches PLAN.md SS11.c):
  normalize -> extract -> [branch: followup? chitchat? fresh need?]
  -> (stage_a ->) [branch: no match? too vague? proceed?]
  -> (clarify | stage_b | stage_b_followup) -> generate -> END

Deliberately NOT built here (later phases): rural-query normalization
beyond a basic strip (Phase 4), a persistent (non-in-memory) session
store, and any web/WhatsApp transport.
"""

import logging
import re
from datetime import date
from typing import Literal, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError, model_validator

from app.core.config import OPENROUTER_API_KEY, OPENROUTER_MODEL
from app.graph.state import ConvState
from app.services.rag_service import RagService, CitizenContext, OCCUPATION_KEYWORDS, eligibility_verdict
from app.services.job_service import JobService, is_under_qualified_for_all
from app.utils import pii_utils
from app.integrations.janaadhaar.service import get_janaadhaar_service

# Same logger name as app/main.py's - both write to one interleaved stream.
logger = logging.getLogger("jan_seva_chat")

KNOWN_CATEGORIES = [
    "Education & Learning",
    "Social welfare & Empowerment",
    "Agriculture,Rural & Environment",
    "Women and Child",
    "Skills & Employment",
    "Health & Wellness",
    "Business & Entrepreneurship",
    "Housing & Shelter",
    "Banking,Financial Services and Insurance",
    "Sports & Culture",
    "Travel & Tourism",
]
KNOWN_BENEFIT_TYPES = [
    "scholarship", "pension", "loan", "subsidy", "equipment",
    "training", "insurance", "housing", "healthcare",
]

FIELD_TYPES_BY_INTENT = {
    "how_to_apply": ["application", "documents"],
    "documents_question": ["documents"],
    # "exclusions" added 2026-08-19 - found live: a scheme's disqualifying/
    # termination conditions (e.g. Krishak Samman Pension Yojana's "if the
    # farmer is out of the small/marginal land category, pension stops")
    # live in this separate field_type, not "eligibility" - without it, the
    # model has no way to know a fact like landholding size even MATTERS,
    # let alone ask about it, since that text was simply never retrieved
    # into its prompt at all for a normal "suggest me a scheme" query.
    "ask_detail": ["eligibility", "benefits", "exclusions"],
    "find_scheme": ["eligibility", "benefits", "exclusions"],
}

# If the need is unspecific (no category/benefit implied) AND this many or
# more schemes survive the eligibility filter, asking one narrowing
# question beats guessing among that many plausible options.
CLARIFY_THRESHOLD = 6

# Cap on how many total clarify_identity_node rounds one need can go through
# (2026-08-24) - the identity-gap check itself is self-terminating (a gap
# stops appearing once its fact is confirmed), so this exists only to bound
# a citizen who never actually answers it (dodges, chitchats around it,
# etc.), not to limit legitimate back-to-back gaps. Set to 3, not 1: a real
# conversation can genuinely need this many rounds (e.g. an
# occupation_target_group gap - disability/transgender/Sahariya-tribal -
# then a caste gap once that's ruled out, then an education_level gap once
# caste narrows the shortlist further) without anything having gone wrong.
MAX_IDENTITY_CLARIFY_ATTEMPTS = 3

# Cap on how many total clarify_jobs_node rounds one need can go through
# (2026-08-26) - was hard-capped at 1 (any prior round at all -> always
# generate_jobs, no matter what), the same over-eager-cap bug class as
# MAX_IDENTITY_CLARIFY_ATTEMPTS above. Found live: "B.Tech... job dekh raha
# hu" (vague) correctly asked about FIELD first (round 1), but once that
# round existed, a still-broad result (5 different titles across 5
# different cities) after the citizen answered "computer science" got
# dumped as a full list instead of also asking about location - a
# genuinely different, still-unresolved narrowing dimension the cap didn't
# distinguish from "asked the same thing twice". Set to 2, matching the
# realistic max depth this domain needs: field/qualification, then
# location - not unbounded, still a real cap.
MAX_JOBS_CLARIFY_ATTEMPTS = 2

# Cap on how many total vague-need clarify_node rounds one need can go through
# (2026-09-02). Was a bare `clarify_attempts < 1` - ONE narrowing question per
# need, ever, no matter what the citizen's answer actually resolved. That is
# the SAME over-eager-cap bug class as the two constants above, and this is now
# its third instance: raised 1 -> 3 for identity gaps (2026-08-24), 1 -> 2 for
# jobs (2026-08-26), while this branch kept the original hard cap.
#
# Found live: "mai kisan hu mere paas 5 acre zameen hai mere liye koi yojna
# btao" correctly asked what KIND of help was wanted (round 1). The citizen
# answered, that answer still left several genuinely different farming schemes
# in play - and because one round had now been spent, the branch was dead, so
# the next turn named a single scheme outright instead of asking the one
# remaining question that actually separated the candidates. The bot then
# asked that very question a turn LATER, unprompted, proving it knew the
# ambiguity was still there and simply had no route left to ask about it.
#
# The 2026-08-10 incident the original cap was written for is a DIFFERENT
# problem: a citizen repeating the same vague answer and getting the identical
# question back verbatim with no way out. That is "asked the same thing twice",
# not "asked a second, genuinely different narrowing question" - the cap could
# not tell them apart, exactly as the jobs constant's comment describes. Set to
# 2, matching the realistic depth this domain needs (purpose, then one further
# axis once purpose is known) - still a real, bounded cap, and generate_node's
# own generate_clarify_rounds HARD STOP remains the final backstop underneath.
MAX_NEED_CLARIFY_ATTEMPTS = 2

# Every scalar identity/eligibility dimension eligibility_verdict already
# checks - not a hand-picked subset added one bug at a time. Deliberately
# NOT grown reactively (gender -> caste -> marital_status -> ...): every
# scheme, present or future, gets mapped into this SAME fixed schema by
# scripts/pipeline/extract_eligibility.py, so this list is already complete by
# construction rather than needing an edit each time a new restriction type
# happens to cause a bug (found live 2026-08-10, after a disability-only-
# restricted scheme slipped through because "disability" wasn't on an
# earlier, hand-picked version of this list - see occupation_target_group's
# handling in _identity_gap for how that specific case is actually covered).
#
# Fail-open correctly keeps a narrowly-restricted scheme (e.g.
# gender=transgender) in the candidate pool when nothing rules it out - but
# that's a different question from whether it should be CONFIDENTLY
# PRESENTED as the answer when nothing confirms it either. Found live
# (2026-08-05): "I want to build my own startup" (no gender stated)
# surfaced "Transgender Utthan Kosh - Self Employment Programme" as the top
# pick - not wrong per the filter, but presenting a narrowly-targeted
# scheme as the confident answer to an unconfirmed identity is the kind of
# ambiguity worth resolving by asking, not guessing.
IDENTITY_FIELDS = ("gender", "caste", "marital_status", "residence", "education_level")

# The two NUMERIC dimensions eligibility_verdict also filters on (age against
# age_min/age_max, family_income against family_income_ceiling) - handled
# separately below from IDENTITY_FIELDS' plain string-equality check, since a
# numeric restriction is a RANGE, not a value to match.
#
# Added 2026-09-01, after tracing why "the bot names a scheme instead of
# asking" kept recurring across unrelated personas (farmer land/pension,
# scholarship income, senior pension) rather than being one persona's bug.
# The root cause was an asymmetry, not a wording problem: eligibility_verdict
# ENFORCES age and family_income whenever the citizen's value is known, but
# _identity_gap never listed them, so when the value was UNKNOWN nothing ever
# asked for it. Every other unconfirmed restriction had a mechanical path to a
# question (clarify_identity_node); these two had none at all, leaving them
# entirely to generate_node's discretion - which is a prompt hint, not a
# guarantee, and one the model routinely spent on a caveat instead ("this fits
# you - but if your land is under 1.5 hectare it won't, let me know").
#
# Measured against the real corpus (data/processed/eligibility_structured.json,
# 157 schemes): 80 of them - 51% - have a genuinely narrow age window, and 21
# have a family_income_ceiling. So this wasn't an edge case; for over half the
# corpus the single most decisive eligibility fact could never be asked about.
#
# IDENTITY_FIELDS' own comment above claims the list is "complete by
# construction ... every scalar identity/eligibility dimension
# eligibility_verdict already checks". That claim was simply no longer true -
# age/income checks were added to eligibility_verdict later and this list was
# never extended to match. These constants restore that stated invariant.
_AGE_GAP_BAND = (18, 80)
# A window only counts as a gap if it's genuinely NARROW. Most schemes are
# tagged 18-100, which is just "any adult" - treating that as a gap would ask
# every citizen their age for no benefit, and (worse) make _first_identity_gap
# see the whole top-5 as gapped almost always, which would fire a clarifying
# question on nearly every turn. Narrowness = what fraction of the 18-80 adult
# band the window actually covers. Verified against the corpus: the split is
# identical at 0.80, 0.85 and 0.90 (80 narrow / 77 wide at all three), i.e.
# there's a clean gap in the data here, not an arbitrary cutoff - the "wide"
# bucket is exactly the any-adult windows (18-100, 1-100, 0-100, 16-100,
# 18-80, 14-100, 20-100, 21-100) and nothing else.
_AGE_WINDOW_NARROW_THRESHOLD = 0.85


def _age_window_is_narrow(meta):
    """Whether this scheme's age window actually excludes a meaningful part of
    the adult population (see _AGE_WINDOW_NARROW_THRESHOLD) - i.e. whether
    knowing the citizen's age would plausibly change the verdict, rather than
    being a formality that admits nearly everyone."""
    lo, hi = meta.get("age_min"), meta.get("age_max")
    lo = 0 if lo is None else lo
    hi = 120 if hi is None else hi
    band_lo, band_hi = _AGE_GAP_BAND
    covered = max(0, min(band_hi, hi) - max(band_lo, lo))
    return covered / (band_hi - band_lo) < _AGE_WINDOW_NARROW_THRESHOLD


def _describe_age_window(meta):
    """Human-readable form of the window, for clarify_identity_node's gap_desc.
    age_max of 100 is this corpus's "no upper limit" sentinel (45 schemes are
    tagged 18-100), not a literal cutoff - phrasing it as "60 to 100 years
    old" would read as though centenarians were being deliberately excluded."""
    lo, hi = meta.get("age_min"), meta.get("age_max")
    lo = 0 if lo is None else lo
    hi = 120 if hi is None else hi
    bounded_lo, bounded_hi = lo > 0, hi < 100
    if bounded_lo and bounded_hi:
        return f"{lo} to {hi} years old"
    if bounded_lo:
        return f"{lo} years or older"
    if bounded_hi:
        return f"{hi} years or younger"
    return "a specific age range"

# How many of the top-ranked schemes get checked for an identity gap, not
# just the #1 - found live (2026-08-10): a #1 scheme unrestricted on gender
# let a #2-ranked, gender-restricted scheme (Transgender Utthan Kosh) slip
# through with no check at all, since only shortlist[0] was ever examined.
# Raised from 3 to 5 (2026-08-12): "start a business" still opened with a
# disability question even after the gap-free-alternative check existed,
# because THREE different narrow-population schemes (differently_abled,
# woman, transgender) happened to rank #1-#3, pushing a genuinely open
# scheme to #4 - just outside the old window. _prioritize_gap_free (which
# decides which schemes to actually SHOW once no question is asked) already
# scans the full shortlist with no such limit; this only widens how deep
# the "is there a gap-free alternative at all" check looks before giving up.
IDENTITY_GAP_CHECK_TOP_N = 5

# Bounds on the rolling context mechanisms below - bounded so prompts don't
# grow without limit, but the history window is deliberately generous:
# raised 4 -> 12 (2026-08-18) after live conversations showed facts stated
# a few turns earlier scrolling out of the window and getting re-asked
# (see ConvState.generate_clarify_rounds' 2026-08-17 seven-questions
# incident). Twelve exchanges of chat text is still a small prompt for the
# models in use; facts that must outlive even this window belong in
# `profile` or `conversation_notes`, which never expire.
MAX_CACHED_CHUNKS = 15
MAX_HISTORY_TURNS = 12
# Cap on the free-form durable memory (ConvState.conversation_notes) -
# oldest notes drop first once exceeded. High enough that a normal
# conversation never hits it.
MAX_CONVERSATION_NOTES = 30

DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")


def _merge_notes(existing, new):
    """Append new free-form notes onto the durable conversation_notes list -
    deduped case-insensitively, oldest dropped first past the cap. Shared by
    both capture channels (extract_profile_node's new_notes and
    extract_need_node's stated_facts) so they feed one list identically."""
    notes = list(existing or [])
    seen = {n.strip().lower() for n in notes}
    for note in new or []:
        cleaned = note.strip()
        if cleaned and cleaned.lower() not in seen:
            notes.append(cleaned)
            seen.add(cleaned.lower())
    return notes[-MAX_CONVERSATION_NOTES:]


def _chunk_key(meta):
    return (meta.get("scheme_id"), meta.get("field_type"), meta.get("lang"))


def _update_chunk_cache(existing_cache, new_chunks, max_size=MAX_CACHED_CHUNKS):
    """Append new_chunks to existing_cache, deduped by (scheme_id, field_type,
    lang) with the newest occurrence winning position (so a re-fetched chunk
    moves to "most recently touched" rather than duplicating), then keep only
    the most recent max_size entries."""
    by_key = {}
    for doc, meta in existing_cache or []:
        by_key[_chunk_key(meta)] = (doc, meta)
    for doc, meta in new_chunks or []:
        by_key[_chunk_key(meta)] = (doc, meta)  # overwrite/reinsert -> moves to end
    return list(by_key.values())[-max_size:]


# Which gap to ask about FIRST when one scheme has several - see the sort at
# the end of _identity_gap. Mirrors generate_node's own QUESTION PRIORITY
# wording ("Prefer broad, low-sensitivity questions (age, class/degree,
# income, residence) over narrow or sensitive ones (disability, caste,
# marital status, gender)"), with the same sensitive sets used by
# _has_sensitive_gap below - deliberately the SAME classification in both
# places rather than a second, drifting copy of it.
_GAP_RANK_BROAD = 0
_GAP_RANK_OTHER = 1
_GAP_RANK_SENSITIVE = 2
_BROAD_GAP_FIELDS = ("age", "family_income", "education_level", "residence")


def _gap_sort_rank(gap):
    field_name, value = gap
    if field_name in _BROAD_GAP_FIELDS:
        return _GAP_RANK_BROAD
    if field_name in _SENSITIVE_IDENTITY_FIELDS:
        return _GAP_RANK_SENSITIVE
    if field_name == "occupation_target_group":
        groups = value.split("|")
        if any(g in _SENSITIVE_OCCUPATION_GROUPS for g in groups):
            return _GAP_RANK_SENSITIVE
        return _GAP_RANK_OTHER
    return _GAP_RANK_OTHER


def _identity_gap(scheme_meta, facts, need_text=""):
    """Which restrictions does this scheme have that current known facts
    haven't confirmed? Returns [(field, required_value), ...].

    Walks the full fixed IDENTITY_FIELDS schema, not a hand-picked subset.
    occupation_target_group is handled separately from the scalar fields
    above: it's usually already implied by what the need itself asks for
    (farm-equipment -> farmer, scholarship -> student), so it's only a real
    gap when the need text doesn't already confirm it - checked via the
    SAME OCCUPATION_KEYWORDS mapping extraction uses to derive this facet in
    the first place, not a new list. Without this exception, every farmer/
    student/entrepreneur-restricted scheme (the majority of the corpus)
    would trigger a redundant clarifying question the need already answered.
    """
    gaps = []
    for field_name in IDENTITY_FIELDS:
        value = scheme_meta.get(field_name, "")
        if value and value != "all" and not facts.get(field_name):
            gaps.append((field_name, value))

    # Numeric dimensions - see _AGE_GAP_BAND's comment for why these were
    # missing entirely until 2026-09-01. `is None` rather than a falsy check
    # for age: a stated age of 0 is meaningless here, but family_income of 0
    # is a real, decisive answer ("koi income nahi hai") that must count as
    # confirmed, not be re-asked forever as if never stated.
    if facts.get("age") is None and _age_window_is_narrow(scheme_meta):
        gaps.append(("age", _describe_age_window(scheme_meta)))

    income_ceiling = scheme_meta.get("family_income_ceiling")
    if income_ceiling is not None and facts.get("family_income") is None:
        gaps.append(("family_income", f"annual family income at or below Rs {income_ceiling}"))

    occ_groups = [g for g in (scheme_meta.get("occupation_target_group") or "").split("|") if g]
    if occ_groups:
        # Check each tag independently, not "is ANY of them confirmed" - a
        # scheme tagged both 'differently_abled' and 'entrepreneur' must not
        # have the sensitive tag waved through just because a mundane one
        # ("business" -> entrepreneur) happens to match the need text too
        # (found live 2026-08-10 testing this exact fix: "start a business"
        # wrongly cleared a disability-restricted scheme's whole gap check).
        #
        # A group already confirmed on the citizen's PERSISTED profile
        # (facts.occupation_groups - see ProfileExtraction) counts as
        # confirmed regardless of what THIS turn's need_text says - added
        # 2026-08-17 because the need_text-keyword check alone only "worked"
        # by coincidence (e.g. a scheme literally named "Krishi Yantra"
        # contains the keyword "krishi") and broke the moment the same
        # citizen's farmer status carried over to a differently-named scheme
        # ("Walk In Tunnel") whose own need_text no longer happened to
        # contain a farming keyword. The keyword-in-need_text check is kept
        # as a fallback for the common single-turn case (the very first
        # message already implies the occupation before profile extraction
        # has even run yet).
        known_groups = set(facts.get("occupation_groups") or [])
        need_lower = (need_text or "").lower()
        unconfirmed_groups = [
            group for group in occ_groups
            if group not in known_groups
            and not any(kw in need_lower for kw in OCCUPATION_KEYWORDS.get(group, []))
        ]
        if unconfirmed_groups:
            gaps.append(("occupation_target_group", "|".join(unconfirmed_groups)))

    # Broad gaps first (2026-09-01). clarify_identity_node only ever asks
    # about gaps[:2], and until now this list came back in raw IDENTITY_FIELDS
    # order - gender, caste, marital_status, residence, education_level -
    # which puts the three SENSITIVE fields at the front and silently drops
    # everything after them. So a scheme gapped on gender + caste + age got
    # asked about gender and caste, and its age requirement (often the fact
    # that actually decides the verdict) was never reachable at all.
    #
    # That directly contradicts the priority both generate_node's prompt and
    # clarify_node's own comment already state: "Prefer broad, low-sensitivity
    # questions (age, class/degree, income, residence) over narrow or
    # sensitive ones (disability, caste, marital status, gender)". The
    # _prioritize_sensitive_gap_free helper below enforces that preference for
    # WHICH SCHEME to ask about, but nothing enforced it for WHICH GAP within
    # the chosen scheme - so the rule was only half-implemented.
    #
    # Stable sort, so ordering within a rank stays exactly as before.
    gaps.sort(key=_gap_sort_rank)
    return gaps


def _first_identity_gap(shortlist, facts, need_text, top_n=IDENTITY_GAP_CHECK_TOP_N):
    """Check the top N shown schemes, not just the top 1, for an unconfirmed
    identity gap - returns (scheme_entry, gaps) for the first one found, or
    (None, []) if none of them have one.

    Only asks when EVERY candidate in the top N has a gap - if even one is
    already usable with no clarification needed, that one wins silently
    rather than blocking on a DIFFERENT candidate's narrower restriction.
    Found live (2026-08-11): "apna business start karna hai" kept opening
    with "are you disabled?" because a disability-gated scheme (occupation_
    target_group='differently_abled') happened to rank #1, even though
    #2/#3/#5 in the same shortlist were ordinary, ungated business schemes
    ready to present with no question at all. Asking about the rank-1
    candidate's gap regardless of what else is available meant a citizen
    never even reached the schemes that already fit them. Rank still
    matters when NOTHING is gap-free (the top-ranked gap is asked about,
    same as before) - this only changes behavior when a clean alternative
    genuinely exists.
    """
    gapped = []
    for entry in shortlist[:top_n]:
        gaps = _identity_gap(entry["metadata"], facts, need_text)
        if gaps:
            gapped.append((entry, gaps))
        else:
            return None, []  # a gap-free candidate exists - nothing to ask about
    if gapped:
        return gapped[0]
    return None, []


def _prioritize_gap_free(shortlist, facts, need_text, limit):
    """Same principle as _first_identity_gap, applied to WHICH schemes get
    explained once no clarifying question is asked - a narrow/rare-
    population restriction (disability, widow, ex-servicemen...) shouldn't
    dominate the citizen's first answer just because it happened to rank
    highest. Gap-free schemes fill the explained slots first; a gapped one
    only fills a remaining slot if there aren't enough gap-free candidates.
    Found live (2026-08-11): even after _first_identity_gap correctly
    stopped blocking on "are you disabled?", stage_b_node was still
    fetching the SAME disability-gated scheme (still ranked #1) as one of
    its top 3 to explain, so generate_node's own prose still centered the
    whole reply around disability instead of the broadly-applicable
    schemes sitting right alongside it."""
    gap_free, gapped = [], []
    for entry in shortlist:
        (gap_free if not _identity_gap(entry["metadata"], facts, need_text) else gapped).append(entry)
    return (gap_free + gapped)[:limit]


# Which of _identity_gap's possible gaps are actually SENSITIVE to ask about
# - matches generate_node's own QUESTION PRIORITY wording verbatim ("narrow
# or sensitive ones: disability, caste, marital status, gender" vs "broad,
# low-sensitivity ones: age, class/degree, income, residence"). _identity_gap
# itself doesn't make this distinction - residence and education_level count
# as a "gap" there exactly the same as caste or disability, just because
# IDENTITY_FIELDS treats all five scalar fields alike for the narrower
# purpose it was built for (does ANY restriction remain unconfirmed at all).
_SENSITIVE_IDENTITY_FIELDS = {"gender", "caste", "marital_status"}
_SENSITIVE_OCCUPATION_GROUPS = {"differently_abled", "widow", "transgender"}


def _has_sensitive_gap(scheme_meta, facts, need_text=""):
    for field_name, value in _identity_gap(scheme_meta, facts, need_text):
        if field_name in _SENSITIVE_IDENTITY_FIELDS:
            return True
        if field_name == "occupation_target_group" and any(g in _SENSITIVE_OCCUPATION_GROUPS for g in value.split("|")):
            return True
    return False


def _prioritize_sensitive_gap_free(shortlist, facts, need_text, limit):
    """Same reordering idea as _prioritize_gap_free, but only demotes a
    scheme gapped on a SENSITIVE field/group - one gapped only on a broad
    fact (education_level, residence) stays in its normal rank position
    instead of being lumped in with caste/disability-gated schemes.

    Needed because _prioritize_gap_free's plain gap-free/gapped split goes
    empty - and its reordering becomes a no-op - the moment every candidate
    has ANY gap at all, sensitive or not. Found live (2026-08-24): for
    "mujhe scholarship chahiye" with no education_level stated yet, EVERY
    one of the top 10 shortlisted scholarships had at least an
    education_level gap (school vs higher-ed unconfirmed), so
    _prioritize_gap_free left the raw rank order untouched and a
    disability-gated scheme sitting at rank 8 was still visible in
    clarify_node's options list - which the model then picked as the
    cleanest narrowing axis, despite being told to prefer a broad one.

    A THREE-tier variant (fully gap-free -> broad-gapped -> sensitive-gapped)
    was tried on 2026-09-02 and REVERTED the same day - do not re-attempt it
    without reading this. The idea was to enforce the other half of
    _first_identity_gap's bargain: that helper stays silent when any top-N
    candidate is gap-free ("a clean option exists, show that"), yet nothing
    stopped stage_b_node presenting a GAPPED scheme anyway. The tier-0
    promotion is relevance-blind, and that turns out to be fatal: since
    _identity_gap started covering age/family_income (2026-09-01, 80 and 21
    schemes respectively), nearly every scheme that MATCHES a real need
    carries some routine gap, while a generic, unrestricted scheme carries
    none - so the gap-free tier systematically promotes exactly the schemes
    with the weakest semantic match. Found live within a day: "mai solar ka
    business start karna chahta hu" ranked "Mukhyamantri Hunar Vikas Yojana -
    Self-Employment/Business" first, but the only fully gap-free candidate in
    the top 5 was "Assistance for Digitization of Business Processes and
    Adoption of eCommerce" (an unrelated MSME e-commerce reimbursement),
    which the tier promoted from rank 5 to rank 1 and got presented as the
    answer. It also did NOT fix the case it was written for (verified: every
    candidate in that scholarship shortlist had an age gap, so tier 0 was
    empty and the order was unchanged) - a strict net loss. Eligibility
    cleanliness belongs as a TIE-BREAKER among comparably-relevant schemes,
    never as a primary sort key over retrieval rank; the sensitive-only
    demotion below is deliberately narrow for exactly that reason."""
    clean, sensitive = [], []
    for entry in shortlist:
        (sensitive if _has_sensitive_gap(entry["metadata"], facts, need_text) else clean).append(entry)
    return (clean + sensitive)[:limit]


def _subject_facts_for_need(need, profile):
    """Whose facts should filter/gate THIS need: the citizen's own persistent
    profile, or the family member's facts stated for this specific need?
    Added 2026-08-10 - a need explicitly asked on behalf of someone else
    (e.g. a son's PhD scholarship) must be judged against THAT person's
    stated facts, never the citizen's own profile, and never left with no
    facts at all just because we correctly stopped attributing them to the
    citizen (see ProfileExtraction's third-party guardrail)."""
    if need and need.get("about") == "family_member":
        return need.get("subject_facts") or {}
    return profile or {}


_llm = None
_reply_llm = None
_rag_service = None
_job_service = None


def get_llm():
    """Extraction/classification calls - temperature 0 so profile facts,
    need classification, and routing stay deterministic and reproducible.

    Every .with_structured_output(...) call anywhere in this file passes
    method="function_calling" explicitly (2026-08-19, found live switching
    OPENROUTER_MODEL from Haiku 4.5 to Sonnet 5) - the default method tries
    strict JSON-schema-constrained decoding, which OpenRouter's OpenAI-
    compatible endpoint doesn't reliably enforce for every model/provider
    behind it. Sonnet 5 returned a discursive markdown answer instead of
    bare JSON for a schema-constrained call and failed Pydantic validation,
    while Haiku 4.5 happened to comply. Forcing function_calling instead
    (tool-calling - the mechanism Claude models support natively and most
    reliably, regardless of which one is configured) makes this robust to
    swapping OPENROUTER_MODEL again later without silently breaking every
    structured-output call in the app."""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=OPENROUTER_MODEL,
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
            # Without a cap OpenRouter reserves the model's full output limit
            # (64k tokens) per call, and rejects the call with 402 once the key's
            # remaining credit is below that - even though replies are short.
            max_tokens=2048,
        )
    return _llm


def get_reply_llm():
    """Citizen-facing reply composition - separate from get_llm() above
    (2026-08-18): temperature 0 on the reply calls made every answer open
    with the identical acknowledgment formula turn after turn ("haan
    samjha..."), reading robotic in live testing. A moderate temperature
    varies phrasing naturally without affecting factual grounding (the
    facts come from the retrieved data in the prompt, not from sampling),
    while extraction/classification stays pinned at 0 where determinism
    actually matters."""
    global _reply_llm
    if _reply_llm is None:
        _reply_llm = ChatOpenAI(
            model=OPENROUTER_MODEL,
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.6,
            # Without a cap OpenRouter reserves the model's full output limit
            # (64k tokens) per call, and rejects the call with 402 once the key's
            # remaining credit is below that - even though replies are short.
            max_tokens=2048,
        )
    return _reply_llm


def _invoke_structured(llm, model_cls, messages, label=""):
    """Wraps a .with_structured_output(model_cls, method="function_calling")
    call with ONE retry on a Pydantic validation error, feeding the exact
    error text back to the model.

    Found live (2026-08-25, crashed to "takniki dikkat" every time): a
    citizen said "...koi aur schemes available hain?" and the model filled
    NeedExtraction.intent with 'wants_alternatives' - a value that's only
    valid for the SIBLING field turn_type, not intent. Every structured-
    output call in this file used to call .invoke() completely unprotected,
    so this ValidationError propagated all the way up through the graph to
    main.py's generic exception handler with no chance to recover - the
    FIRST validation failure of any kind, on ANY of these calls, crashed
    the whole turn. Showing the model its own exact error text usually
    self-corrects a mistake like this, since it's legible once pointed
    out - the same "point at the specific violation, don't just restate
    the rule" principle already proven for generate_node's leaked-name
    retry. If the retry ALSO fails, this re-raises - main.py's handler is
    still the final backstop, just no longer the FIRST line of defense."""
    structured_llm = llm.with_structured_output(model_cls, method="function_calling")
    try:
        return structured_llm.invoke(messages)
    except ValidationError as e:
        logger.warning("(%s) structured-output validation failed, retrying once: %s", label, e)
        correction = (
            "Your previous attempt failed schema validation with this exact error - re-read which "
            f"FIELD each value belongs to and try again:\n{e}"
        )
        if isinstance(messages, str):
            retry_messages = f"{messages}\n\n{correction}"
        else:
            retry_messages = list(messages) + [("human", correction)]
        return structured_llm.invoke(retry_messages)


# Fallback text for _invoke_reply's own failure case - deliberately generic
# (not scheme/job-specific) since it's used from every plain reply-composing
# node, and reused across languages the same way other fixed texts in this
# file are (_ASK_CONSENT_TEXT etc.).
_REPLY_FAILED_TEXT = {
    "hindi": "माफ़ कीजिए, जवाब देते समय कुछ दिक्कत आ गई। कृपया अपना सवाल दोबारा पूछें।",
    "hinglish": "Maaf kijiye, jawab dete waqt kuch dikkat aa gayi. Kripya apna sawaal dobara poochhein.",
    "english": "Sorry, something went wrong composing a reply. Please ask your question again.",
}


def _invoke_reply(llm, messages, reply_lang, label=""):
    """Wraps a plain (non-structured) get_reply_llm().invoke(messages).content
    call - catches ANY exception (network/timeout/rate-limit/auth error from
    OpenRouter, not just the ValidationError _invoke_structured guards
    against) and returns an honest, per-language fallback reply instead of
    letting it crash the whole turn.

    Found live (2026-08-25 audit): clarify_node/clarify_jobs_node/
    clarify_identity_node/chitchat_node/generate_jobs_node/
    _answer_from_entitlements all called get_reply_llm().invoke(...).content
    completely unprotected - the worst concrete case, a citizen who's JUST
    successfully validated a Jan Aadhaar OTP hits a transient failure
    composing the final answer; the OTP is already consumed but the turn
    crashes before that gets recorded, leaving them stuck re-entering a
    dead OTP with no way forward except restarting verification."""
    try:
        return llm.invoke(messages).content
    except Exception as e:
        logger.warning("(%s) plain reply call failed: %s", label, e)
        return _pick_text(reply_lang, _REPLY_FAILED_TEXT)


# Shared persona for every citizen-facing reply call, sent as the system
# message (2026-08-18). Before this, each reply node built a bare one-off
# instruction prompt with no voice at all, and the deterministic model fell
# back on the same stock acknowledgments every turn - a live complaint
# ("sounds robotic, says 'haan samjha' again and again") distinct from any
# single prompt rule being wrong. Content/grounding rules stay in each
# node's own prompt; this only sets tone.
PERSONA = (
    "You are Jan Seva Sahayak, a warm, friendly assistant helping Rajasthan citizens find "
    "government welfare schemes and job vacancies. Talk the way a helpful, knowledgeable friend "
    "or relative would explain this over chai - natural, warm, a little casual - never like a "
    "form being filled in or a script being read out. Vary your wording naturally from reply to "
    "reply: do not open consecutive replies with the same acknowledgment phrase (no stock "
    "'haan samjha' / 'theek hai' / 'sure' openers repeated turn after turn - usually just get "
    "to the point), and do not restate back what the citizen just told you. Never re-ask "
    "something the citizen has already answered anywhere in the conversation context you are "
    "given.\n\n"
    "FORMATTING: break up anything with more than one piece of information into a short "
    "markdown heading (## or ###) or **bold** key terms/numbers, plus a bullet list - not one "
    "dense paragraph. This makes multi-scheme or multi-step answers easy to scan at a glance, "
    "same as a knowledgeable person would sketch it out rather than read it out loud. Do NOT use "
    "markdown tables (pipe-and-dash grids) - they render as broken text here; use a bullet list "
    "instead for anything that would otherwise be a table. A one-line answer doesn't need any of "
    "this structure - use it only when it actually helps organize more than a sentence or two.\n\n"
    "TONE: match the citizen's own register. In Hindi/Hinglish, it's fine to sound like normal "
    "warm conversational speech (dekho, chalo, arre, ek minute) rather than stiff textbook "
    "Hindi - the goal is to sound like a person, not a government notice. A friendly emoji here "
    "and there is fine when it fits naturally, not in every reply. Keep replies concise and "
    "practical either way - warmth and structure are about making it easy to read, not about "
    "making it longer.\n\n"
    "Casual does NOT mean commanding - whenever you ask the citizen something, phrase it as a "
    "polite request or a stated need, never a bare imperative that reads as an order. Found live "
    "(2026-08-18): 'pehle batao' ('tell me first') read as rude/blunt even though the rest of the "
    "reply was warm. Prefer 'mujhe ek baat jaanani hai', 'ek baat bataiye/batayenge', 'mujhe kuch "
    "jankari chahiye' - a request or a stated need, not a command - same idea in English: 'could "
    "you tell me' over a flat 'tell me'.\n\n"
    "Casual also does NOT mean substituting your own blunter word for a family member the citizen "
    "mentioned - always mirror the EXACT term the citizen themselves used (pitaji, papa, maa, "
    "ammi, bhai, behen, whatever word they chose), never swap in a cruder synonym of your own even "
    "if it's common slang elsewhere. Found live (2026-08-18): the citizen said 'mere pitaji' (a "
    "respectful term) and the reply said 'aapke baap' - a blunt, disrespectful word for the exact "
    "same person, purely because it's casual - respect matters more than casualness whenever "
    "the two pull in different directions."
)


def get_rag_service():
    global _rag_service
    if _rag_service is None:
        _rag_service = RagService()
    return _rag_service


def get_job_service():
    global _job_service
    if _job_service is None:
        _job_service = JobService()
    return _job_service


Intent = Literal["find_scheme", "ask_detail", "how_to_apply", "documents_question", "chitchat"]
YesNo = Literal["yes", "no"]
RuralUrban = Literal["rural", "urban"]
EducationLevel = Literal["school", "higher_education"]
# Three-way, not the old Devanagari-detected hi/en binary (app/graph/chat.py's old
# normalize_node) - that binary silently bucketed Hinglish (Romanized Hindi)
# in with English, so a citizen writing "mujhe scheme chahiye" got replied
# to in prose English instead of matching their own Hinglish register.
# 'hindi' = Devanagari script. 'hinglish' = Hindi/Rajasthani words in Roman
# script (even mixed with occasional English words) - the common rural
# register. 'english' = genuine English vocabulary/grammar, not Romanized
# Hindi.
ReplyLanguage = Literal["hindi", "hinglish", "english"]
# Single mutually-exclusive classification, replacing what used to be two
# independent booleans (is_followup, wants_alternatives) that could - and
# did, live - contradict each other (is_followup=no + wants_alternatives=yes
# for a message that was neither a followup nor wanted alternatives, it was
# a plain new need). One field can't structurally disagree with itself the
# way two independently-generated fields can.
TurnType = Literal["new_need", "followup_same_scheme", "wants_alternatives"]


def _normalize_to_known(value, known_values):
    """Snap free-text LLM output to the closest known canonical string
    (case/whitespace-insensitive substring match), rather than forcing the
    LLM to emit a byte-perfect enum value - Groq's strict tool-call
    validation rejects harmless phrasing variance (e.g. an added space,
    "Agriculture, Rural" vs the stored "Agriculture,Rural") as a hard
    error, so normalization belongs in our own code, not the model's output.
    """
    if not value:
        return None
    squashed = re.sub(r"\s+", "", value).lower()
    for known in known_values:
        if re.sub(r"\s+", "", known).lower() == squashed:
            return known
    for known in known_values:
        if squashed in re.sub(r"\s+", "", known).lower():
            return known
    return None


# Strict, per-language reply instructions (2026-08-11) - replaces the old
# binary "Hindi (Devanagari)"/"English" instruction, which had no Hinglish
# option at all, so a citizen writing in Hinglish got a reply in prose
# English instead of matching their own register. Explicitly calls out
# "no mixing" in both directions, since the failure mode observed wasn't
# just "wrong language" but partial mixing (a Devanagari word inside an
# English sentence, or vice versa). Official scheme names are the one
# deliberate exception in all three - they're proper nouns, not translated.
_LANG_INSTRUCTIONS = {
    # Deliberately allows English words IN Devanagari script (2026-08-18,
    # replacing a stricter "zero English words" rule) - found live: voice
    # input in Hindi always transcribes to Devanagari (that's just how
    # Whisper writes Hindi, regardless of how casually someone actually
    # spoke), which always classified as this register, and the old strict
    # rule made every voice-Hindi reply come out textbook-formal - a citizen
    # typing casually could switch to 'hinglish' (Roman script) to get a
    # warmer register, but a voice user structurally never could. The fix
    # isn't to change which register voice-Hindi maps to (still Devanagari,
    # since that's genuinely what was spoken) - it's to make THIS register
    # itself as natural as real Hindi speakers actually write, mixing in
    # English words/technical terms the way this bot's own reference style
    # already does (e.g. "ERPNext ko 3 layers mein samjho", "Backend "
    # "language: Python" - Devanagari grammar/structure, English words "
    # dropped in naturally, not translated into stiff Hindi equivalents).
    "hindi": (
        "Reply in Hindi, in Devanagari script for the Hindi words/grammar - but write it the way "
        "a real Hindi speaker actually talks, freely mixing in English words for technical terms, "
        "common loanwords, or anything that would sound stiff if translated (e.g. 'Framework', "
        "'Backend', numbers/amounts, official scheme names) rather than forcing a pure/textbook "
        "Hindi equivalent for everything. Do NOT use Romanized Hindi (Hinglish) - the Hindi PARTS "
        "of the reply must stay in Devanagari script; only the mixed-in English words are in Roman "
        "letters, exactly like this reference style (Devanagari for the Hindi grammar, Roman only "
        "for the English terms mixed into it - NOT the fully-Romanized 'ERPNext ko samjho' form): "
        "'ERPNext को समझो - Frappe Framework, Python Backend, MariaDB Database.' Found live "
        "(2026-08-19): an earlier version of this instruction gave a fully-Romanized example here "
        "by mistake, and replies came out entirely in Roman script (Hinglish) despite the rule "
        "above explicitly saying Devanagari - the example was contradicting the rule, and the "
        "example won. Getting this example's OWN script right matters more than the prose rule. "
        "This app supports exactly three registers: Hindi (Devanagari), Hinglish (Roman), and "
        "English - NEVER any other Indian script (Gurmukhi/Punjabi, Bengali, Gujarati, etc.), "
        "even briefly for one word - found live (2026-08-19) a reply came back in Gurmukhi script "
        "with no instruction anywhere calling for that; if Devanagari is called for, it is "
        "Devanagari only, start to finish. A citizen's OWN message using many English loanwords "
        "spelled out in Devanagari (e.g. 'स्कीम्स', 'अवेलेबल', 'हायर एजुकेशन') does NOT license "
        "switching the REPLY to Hinglish/Roman script either - found live (2026-08-26): a message "
        "this loanword-heavy got a full Hinglish reply immediately after two correctly-Devanagari "
        "replies to the same citizen, then flipped straight back to Devanagari the very next turn - "
        "matching the citizen's own vocabulary choice is not the same as matching their script; "
        "keep composing the Hindi grammar/structure in Devanagari regardless of how many individual "
        "words in the citizen's message were English loanwords."
    ),
    "hinglish": (
        "Reply in Hinglish - Hindi words and grammar written in Roman/Latin script, the way a "
        "Hindi speaker naturally types. Same rule as the Hindi register above: freely mix in "
        "English words for technical terms, common loanwords, or anything that sounds stiff "
        "translated (e.g. 'Framework', 'Backend', numbers/amounts, official scheme names), same "
        "as this reference style: 'ERPNext ko samjho - Frappe Framework, Python Backend, MariaDB "
        "Database.' Do NOT use Devanagari script anywhere - everything (Hindi parts AND the "
        "mixed-in English words) stays in Roman/Latin letters. Do not write in stiff, fully "
        "formal prose English either - match the citizen's own Hinglish register."
    ),
    "english": (
        "Reply ENTIRELY in English. Do not include any Hindi words, in Devanagari or Romanized "
        "form, anywhere in the reply, except official scheme names, which are proper nouns and "
        "stay as-is."
    ),
}

# Current-turn-wins rule for concrete stated details (2026-08-24) - added
# after a live bug: a citizen asked about Jaipur schemes for several turns,
# then switched to "ab Jodhpur ke bare mein batao" - this turn's own
# retrieval correctly used Jodhpur (need_text is rebuilt fresh every turn),
# but the REPLY TEXT still said Jaipur, because conversation_history below
# carries the last MAX_HISTORY_TURNS turns verbatim and "Jaipur" appeared
# across several of them while "Jodhpur" appeared only once, in this turn's
# own question. Nothing told the model which value should win when
# composing prose, so it echoed whichever one had more textual weight in
# the accumulated context rather than the one actually being asked about
# right now. Not specific to city names - the same failure mode can hit any
# concrete stated detail (a number, a place, a person) that changes
# mid-conversation. Only wired into generate_node/generate_jobs_node - the
# two nodes that actually state a concrete fact back to the citizen as
# prose; clarify_node/clarify_jobs_node/clarify_identity_node only ever ask
# a question (never restate a fact as a claim) and extract_profile_node/
# extract_need_node already rebuild need_text/profile fresh from the
# current message each turn, so neither has the failure this rule guards
# against.
_RECENCY_RULE = (
    "RECENCY: if a concrete detail stated THIS turn (in the citizen's current question or exact "
    "words) - a place name, a number, a person, a preference - differs from something mentioned "
    "earlier in the conversation above, THIS TURN's value is authoritative. Answer using the "
    "current one; do not carry forward, restate, or blend in the earlier value, even if it "
    "appears more often in the conversation so far."
)


def _lang_instruction(reply_lang):
    return _LANG_INSTRUCTIONS.get(reply_lang, _LANG_INSTRUCTIONS["english"])


def _source_lang(reply_lang):
    """Which of the corpus's two scraped language versions (en/hi) to pull
    scheme TEXT from - distinct from the reply language instruction above.
    Hinglish reuses the Hindi source text (genuine Hindi phrasing to work
    from) even though the REPLY itself must come out in Roman script, not
    Devanagari - conflating these two would mean fetching English source
    text and asking the model to invent Hindi-flavored phrasing from
    scratch instead of grounding it in the scheme's actual Hindi wording."""
    return "hi" if reply_lang in ("hindi", "hinglish") else "en"


# Keys a provider sometimes wraps the real tool-call arguments in, instead of
# returning them bare. Both are JSON-Schema/OpenAI-tool-envelope names, never
# a field any model below actually declares - so seeing one at the top level
# is unambiguous evidence of a wrapper, not of real content.
_TOOL_ARG_ENVELOPE_KEYS = ("parameters", "arguments")


class _UnwrapsToolEnvelope(BaseModel):
    """Base for every .with_structured_output(...) model in this file - lifts
    a `{"parameters": {...}}` / `{"arguments": {...}}` wrapper up one level
    before validation runs.

    Found live (2026-09-01, in the server log): an extract_need call failed
    with FIVE "Field required" errors at once - needs, need_is_specific,
    intent, turn_type, detected_language - and the logged input_value was
    `{'parameters': {'needs': ... "son is 32 years old"}}`. The model's actual
    content was entirely correct; it had simply returned the arguments still
    wrapped in the JSON-Schema-shaped envelope rather than bare, so Pydantic
    found none of the required fields at the top level and every single one
    reported missing simultaneously. That all-fields-at-once signature is the
    giveaway that it's a serialization mismatch, not a reasoning failure.
    OpenRouter does this intermittently with method="function_calling",
    depending on how the upstream provider re-serializes the tool call.

    Worth fixing deterministically rather than leaving to _invoke_structured's
    retry: the retry only recovered because the model happened to re-emit it
    unwrapped the second time (nothing about the correction text addresses an
    envelope - it just shows the field-missing errors), and it costs a full
    extra call of one of the largest prompts here every time it fires. If the
    retry ALSO wrapped it, _invoke_structured re-raises and the whole turn
    fails to a "takniki dikkat" message. This makes the envelope simply never
    reach validation.

    Only unwraps when the wrapper is the ONLY key present and its value is a
    dict - a real payload that legitimately contained a field called
    "parameters" alongside its other fields is left completely untouched.
    """

    @model_validator(mode="before")
    @classmethod
    def _lift_tool_envelope(cls, data):
        if not isinstance(data, dict) or len(data) != 1:
            return data
        (only_key, only_value), = data.items()
        if only_key in _TOOL_ARG_ENVELOPE_KEYS and isinstance(only_value, dict):
            return only_value
        return data


class NeedItem(_UnwrapsToolEnvelope):
    """One distinct need/topic within the message. A message almost always
    produces exactly one of these - it only becomes a list because a single
    message occasionally states two unrelated needs at once (a farming
    subsidy for the citizen AND a scholarship for their son), and each needs
    its own search + its own eligibility facts rather than being merged or
    silently dropped."""

    need_text: str = Field(
        ...,
        description=(
            "A SHORT (3-6 word) phrase capturing ONLY the core purpose/need - e.g. 'start a "
            "business', 'scholarship for SC student'. Do NOT include place names, institution "
            "names, degree specifics, or other biographical detail (that's captured separately) - "
            "this phrase is used directly as a search query, and extra detail hurts semantic "
            "search relevance rather than helping it (verified 2026-08-05: a query padded with "
            "'IIT Kharagpur'/'B.Tech computer science'/'Jodhpur' surfaced irrelevant agriculture "
            "schemes ahead of real business schemes; the same need stated as just 'start a "
            "company' retrieved correctly).\n"
            "If the citizen wrote in Hindi OR Hinglish (romanized Hindi), write this phrase in "
            "DEVANAGARI SCRIPT (transliterate/translate as needed), even if their own message "
            "used Roman letters. Verified 2026-08-07: 'beti ki shaadi' (Hinglish) retrieved "
            "irrelevant construction-worker schemes, while the same need as 'बेटी की शादी' "
            "(Devanagari) correctly retrieved actual marriage-assistance schemes - the embedding "
            "search matches natural Hindi phrasing far better in Devanagari than in Roman script. "
            "If they wrote in English, keep this phrase in English."
        ),
    )
    need_category: Optional[str] = Field(
        None,
        description=(
            f"one of {KNOWN_CATEGORIES} ONLY if the message itself names or clearly implies a "
            "subject matter. If the message is a pure procedural/reference follow-up with no "
            "subject matter of its own (e.g. 'what's the applying process of this', 'iske "
            "documents batao', 'just give me the process for this') - referring back to "
            "something already discussed rather than describing a new topic - leave this null. "
            "Do NOT guess a category from words like 'apply'/'process'/'documents' alone; those "
            "describe an ACTION, not a subject."
        ),
    )
    need_benefit_type: Optional[str] = Field(
        None,
        description=(
            f"one of {KNOWN_BENEFIT_TYPES} ONLY if the message itself names or clearly implies "
            "one, using the same rule as need_category - null for a content-free procedural "
            "follow-up, do not guess."
        ),
    )
    domain: Literal["scheme", "job", "citizen_status"] = Field(
        ...,
        description=(
            "'job' if this need is about a GOVERNMENT JOB VACANCY/RECRUITMENT - looking for a "
            "sarkari naukri, bharti, vacancy, a specific post (Patwari, LDC, Police Constable, "
            "Junior Assistant, etc.), or asking what jobs/posts they can apply for.\n"
            "'citizen_status' ONLY if the citizen is asking about THEIR OWN existing "
            "record/application/benefit status with the government - e.g. 'kya meri pension "
            "aayegi', 'mera application status kya hai', 'meri scholarship kab aayegi' - a "
            "question that can only be answered by looking up THEM specifically, not a general "
            "eligibility/discovery question. Do NOT use this for 'am I eligible for X' or 'what "
            "schemes can I get' - those are 'scheme', even if phrased personally, since they're "
            "about discovering options, not checking an existing record.\n"
            "'scheme' for everything else - a welfare benefit, subsidy, pension, scholarship, "
            "loan, or other government ASSISTANCE scheme (including business/employment-support "
            "schemes, which are benefits, not a job posting). When genuinely unclear, default to "
            "'scheme'."
        ),
    )
    job_location: Optional[str] = Field(
        None,
        description=(
            "ONLY if domain='job' and the citizen named a specific city for the job search (e.g. "
            "'Jaipur', 'Jodhpur'). null otherwise."
        ),
    )
    job_qualification: Optional[Literal["Below 12th", "12th Pass", "Diploma", "Graduate", "Post Graduate"]] = Field(
        None,
        description=(
            "ONLY if domain='job' and the citizen explicitly stated THEIR OWN educational/technical "
            "qualification - either in THIS message, or recorded in the 'Durable facts noted "
            "earlier' context above (a qualification they stated in an earlier turn still counts; "
            "never make them restate it). Normalize free text to exactly one of these 5 levels: "
            "'Below 12th' (8th/10th pass, or any schooling short of completing 12th); "
            "'12th Pass' (completed 12th/higher secondary, no further study); 'Diploma' "
            "(polytechnic, ITI, any technical diploma); 'Graduate' (any bachelor's degree - "
            "B.A./B.Sc./B.Com/B.Tech/BCA/etc.); 'Post Graduate' (any master's degree or higher). "
            "null if no qualification was stated - do not guess this from their phrasing or tone.\n"
            "'Below 12th' was added 2026-09-02 because it was MISSING, and its absence caused a "
            "real failure rather than just an approximation: this description used to define "
            "'12th Pass' as '(10th/12th pass, no further study)', explicitly instructing that a "
            "10th-pass citizen be recorded as 12th Pass. Found live - 'mai 10th pass hu' was "
            "stored as '12th Pass', so the very next reply opened '12th pass ke liye Rajasthan "
            "mein kaafi vacancies aati hain', silently upgrading the citizen. Worse, the "
            "structured field then CONTRADICTED the conversation text, which still plainly said "
            "10th: the reply ended up both asserting they were 10th pass AND listing 12th-Pass-"
            "only vacancies AND asking a third time whether they were 10th or 12th - an "
            "unresolvable loop caused entirely by there being no honest value to store. Never "
            "round a qualification UP to the nearest available level; if the citizen is below "
            "every level listed here, 'Below 12th' is the correct answer."
        ),
    )
    job_min_salary: Optional[int] = Field(
        None,
        description=(
            "ONLY if domain='job' and the citizen explicitly stated a minimum/expected salary "
            "figure - in THIS message (e.g. '30000 se zyada chahiye', 'at least 25k salary') or "
            "recorded in the 'Durable facts noted earlier' context above. The "
            "number in rupees (convert 'k'/'lakh' as needed, e.g. '25k' -> 25000, '2 lakh' -> "
            "200000). null if no figure was stated."
        ),
    )
    about: Literal["self", "family_member"] = Field(
        ...,
        description=(
            "'family_member' ONLY if this specific need is explicitly stated as being for someone "
            "else the citizen is asking on behalf of (son/daughter/spouse/parent/sibling) - e.g. "
            "'scheme for my beti', 'mera beta PhD karna chahta hai uske liye kuch batao'. 'self' "
            "for everything else, including when the message doesn't specify - default to self."
        ),
    )
    subject_relation: Optional[str] = Field(
        None,
        description="ONLY if about='family_member': a short English word for who (son, daughter, spouse, mother, father, sibling). null if about='self'.",
    )
    subject_caste: Optional[str] = Field(
        None,
        description="ONLY if about='family_member' AND their caste was explicitly stated for THIS need (e.g. sc/st/obc/general/pvtg). null otherwise, and null if about='self' (use the citizen's own profile instead).",
    )
    subject_gender: Optional[str] = Field(
        None,
        description="ONLY if about='family_member' AND their gender was explicitly stated for THIS need. null otherwise.",
    )
    subject_education_level: Optional[EducationLevel] = Field(
        None,
        description=(
            "ONLY if about='family_member': 'school' if they're CURRENTLY STUDYING IN (not yet passed) "
            "school, up to Class 12. 'higher_education' if pursuing/holding a degree, diploma, or PhD - "
            "OR if they've PASSED/COMPLETED Class 12 and are now seeking to continue into further/higher "
            "studies (e.g. '12th pass, wants to study further') - that is 'higher_education' even before "
            "a degree has started, because that is the stage they are now entering. "
            "null if not stated or about='self'."
        ),
    )
    subject_disability: Optional[YesNo] = Field(
        None,
        description=(
            "ONLY if about='family_member': 'yes' if THEY were explicitly stated to have a "
            "disability, 'no' if explicitly stated to NOT have one (denying one specific thing "
            "does not confirm anything else - same rule as the citizen's own disability field "
            "above). null if not stated or about='self'."
        ),
    )
    subject_age: Optional[int] = Field(
        None,
        description="ONLY if about='family_member' AND their age was explicitly stated for THIS need. null otherwise.",
    )
    subject_marital_status: Optional[str] = Field(
        None,
        description=(
            "ONLY if about='family_member': married, never-married, widowed, divorced, separated - "
            "only if explicitly stated about THEM specifically. null if not stated or about='self'."
        ),
    )
    subject_family_income: Optional[int] = Field(
        None,
        description=(
            "ONLY if about='family_member' AND an income figure specific to them (not the general "
            "family income, which belongs on the citizen's own profile) was explicitly stated. "
            "null otherwise - most of the time this stays null even for family-member needs, since "
            "family income is usually a household-level fact already captured on the citizen."
        ),
    )
    subject_residence_evidence: Optional[str] = Field(
        None,
        description=(
            "ONLY if about='family_member': the EXACT words, copied verbatim, that specifically say "
            "WHAT TYPE of area THEY live in (village/rural vs. city/urban) - same rule as the "
            "citizen's own residence_evidence field above: a place name alone does not qualify, "
            "and this is usually null even for family-member needs since residence is normally a "
            "shared household fact already on the citizen's own profile."
        ),
    )
    subject_residence: Optional[RuralUrban] = Field(
        None,
        description="ONLY if about='family_member' and subject_residence_evidence above is also filled in with the literal supporting words. null otherwise.",
    )


class ProfileExtraction(_UnwrapsToolEnvelope):
    """Just the mechanical, low-ambiguity part: did the citizen state a fact
    about THEMSELVES (the person sending this message) this message? Split
    out from need/intent classification (below) after finding that one call
    trying to fill ~10 fields at once was flaky - even outright dropping
    required fields on simple messages like "main mahila hoon". Fewer
    fields per call = fewer chances to fail.

    RE-CONFIRMED (2026-08-11): a later attempt to re-merge this with
    NeedExtraction into one call was tried again and reverted a second
    time - verified directly that it reintroduces a real regression (the
    residence_evidence guardrail below started confidently misclassifying
    'main Rajasthan ka rehne wala hu' as rural evidence, 11/11 across two
    rounds including after strengthening the field description, despite
    the standalone version below getting it right 5/5). Do not re-attempt
    merging this with need classification without re-verifying against
    that exact case first.

    Every field here is about the citizen who is typing, never about a
    family member (spouse/child/parent/sibling) they mention. A message
    like "mujhe beti ke liye scholarship chahiye" (I need a scholarship for
    my daughter) states a fact about the daughter, not the citizen - it
    does NOT mean the citizen is male or married; having a daughter implies
    nothing about the citizen's own gender or marital status. Similarly
    "mera beta 12th mein hai" describes the SON's education, not the
    citizen's own education_level. If a fact in the message clearly belongs
    to someone other than the citizen, leave that field null rather than
    attributing it to the citizen - only fill a field when the citizen is
    describing themselves.
    """

    caste: Optional[str] = Field(None, description="e.g. sc, st, obc, general, pvtg - only if the citizen just stated it about THEMSELVES this message")
    gender_evidence: Optional[str] = Field(
        None,
        description=(
            "The EXACT words, copied verbatim, that indicate the citizen's OWN gender - actively "
            "search for BOTH kinds of evidence before answering: (1) an explicit word - "
            "'mahila'/'aurat'/'female' for female, 'purush'/'aadmi'/'male' for male; OR (2) a "
            "Hindi grammatical marker - a masculine verb form or 'wala'/'wale' suffix in a "
            "first-person statement ('rehta hoon', 'rehne wala hoon', 'karta hoon') is JUST AS "
            "VALID evidence of male as a feminine verb form or 'wali' suffix ('rehti hoon', 'rehne "
            "wali hoon', 'karti hoon') is of female. The masculine grammatical form is NOT an "
            "unmarked default to skip past - it is direct, explicit evidence, exactly as much as "
            "the feminine form is. null if no such words are present anywhere in the message."
        ),
    )
    gender: Optional[str] = Field(
        None,
        description=(
            "male, female, transgender - set this whenever gender_evidence above identifies a "
            "real marker (explicit word or grammatical form, either masculine or feminine). Only "
            "if the citizen just stated their OWN gender, never inferred from a family member "
            "they mention."
        ),
    )
    marital_status: Optional[str] = Field(None, description="married, never-married, widowed, divorced, separated - only if the citizen just stated their OWN marital status, never inferred from mentioning a spouse or child")
    age: Optional[int] = Field(None, description="the citizen's own age - only if just stated")
    family_income: Optional[int] = Field(None, description="annual family income in rupees - only if just stated")
    residence_evidence: Optional[str] = Field(
        None,
        description=(
            "The EXACT words, copied verbatim from the citizen's message, that specifically say "
            "WHAT TYPE of area they live in - a village/countryside, or a city/town. This must go "
            "beyond merely naming a place or stating that they are a resident of somewhere: 'main "
            "Rajasthan ka rehne wala hu' / 'I live in Jaipur' / 'I'm from Delhi' name a PLACE but do "
            "NOT say what type of area it is, so none of those qualify - leave this null for them. "
            "Only a phrase that itself signals village/rural vs. city/urban character qualifies - "
            "e.g. 'gaon mein rehta hoon', 'dehat se hoon', 'main shehar mein rehti hoon', 'I live in "
            "a city'. If you cannot find such a phrase, leave this null even if a place name, "
            "occupation, or other detail is present elsewhere in the message."
        ),
    )
    residence: Optional[RuralUrban] = Field(
        None,
        description=(
            "MUST be exactly 'rural' or 'urban' if they described where THEY live in those terms - "
            "e.g. 'village', 'gaon' -> rural; 'city', 'shehar' -> urban. "
            "If they only named a specific place (a city/district/state name like 'Jaipur'), leave this "
            "null - a place name is NOT a value for this field, we don't track specific locations. "
            "Only set this if residence_evidence above is also filled in with the literal supporting words."
        ),
    )
    education_level: Optional[EducationLevel] = Field(
        None,
        description=(
            "The CITIZEN's own education level - 'school' if they said THEY are CURRENTLY STUDYING IN "
            "(not yet passed) school, up to Class 12 / 10+2. 'higher_education' if they said THEY are a "
            "graduate, post-graduate, currently pursuing/have completed a degree/diploma/professional "
            "course (e.g. 'I'm a BCA graduate', 'MCA graduate', 'pursuing engineering') - OR if they've "
            "PASSED/COMPLETED Class 12 and are now seeking to continue into further/higher studies (e.g. "
            "'12th pass, want to study further') - this is 'higher_education' even though they haven't "
            "started a degree yet, because that is the stage they are now entering; 'school' means still "
            "enrolled in and yet to finish school, not already past it. "
            "If the education level mentioned belongs to their child/sibling/spouse instead of "
            "themselves, leave this null - it is not the citizen's own education_level. "
            "Only if just stated this message."
        ),
    )
    disability: Optional[YesNo] = Field(
        None,
        description=(
            "'yes' ONLY if the citizen explicitly stated THEY THEMSELVES have a disability "
            "(e.g. 'I have 40% disability', 'mujhe viklangta hai'). 'no' ONLY if they explicitly "
            "stated THEY do NOT have a disability (e.g. 'no i dont have any disability'). null if "
            "disability wasn't mentioned about the citizen themselves at all - do not guess from "
            "occupation, need, or anything else."
        ),
    )
    occupation_groups: list[str] = Field(
        default_factory=list,
        description=(
            "Which of these categories did the CITIZEN THEMSELVES (not a family member) just state "
            "about themselves THIS message: farmer, student, widow, differently_abled, unemployed, "
            "senior_citizen, woman, artisan, labourer, entrepreneur, ex_servicemen. Include a "
            "category only when this message itself clearly states or directly implies it about the "
            "citizen (e.g. 'mai kheti karta hu'/'I farm' -> farmer; 'mera business hai'/'I run a "
            "business' -> entrepreneur; 'main padhta hoon' -> student) - not because it's associated "
            "with whatever scheme happens to be under discussion. A citizen can genuinely belong to "
            "more than one (e.g. a farmer who also runs a business) - include every one THIS message "
            "states, not just one. Empty list if nothing new is stated this message - this only ADDS "
            "to what's already known, so previously-confirmed categories never need restating."
        ),
    )
    new_notes: list[str] = Field(
        default_factory=list,
        description=(
            "OTHER durable facts stated in THIS message that are worth remembering for the rest of "
            "the conversation but have NO dedicated field above - e.g. facts about a family member "
            "('son is in class 10', 'daughter is 19 and unmarried'), documents the citizen has or "
            "lacks ('has a BPL card', 'no disability certificate'), schemes/jobs they already "
            "applied to or already receive, land/assets they own ('owns 2 bigha of land'), their "
            "job-search preferences (qualification, expected salary, preferred city), or a direct "
            "answer they just gave to a question in your last reply - INCLUDING negative answers "
            "('said no when asked about disability'). One short, self-contained English sentence "
            "per fact, stating whose fact it is. Only facts actually stated in THIS message - "
            "never inferences - and never a duplicate of something already in the known facts or "
            "already-noted list above. Most messages have none - empty list is the normal case."
        ),
    )


class NeedExtraction(_UnwrapsToolEnvelope):
    """The judgment-heavy part: what do they want, and is it clear enough
    to act on. need_is_specific is the model's OWN direct clarity call -
    never inferred from whether need_category/need_benefit_type happened
    to populate (those two are unreliable and exist only for the rank
    boost, not for judging vagueness).

    Includes chitchat_reply (2026-08-11) - folding the chitchat reply
    itself into this same call (verified safe/reliable, unlike merging in
    ProfileExtraction's fields - see that class's docstring) skips a
    separate chitchat_node call entirely on chitchat turns."""

    needs: list[NeedItem] = Field(
        ...,
        description=(
            "One entry per DISTINCT need/topic in the message. Almost always exactly ONE entry - "
            "only produce more than one when the citizen clearly states two or more SEPARATE, "
            "unrelated purposes in the same message (e.g. a farming subsidy for themselves AND a "
            "scholarship for their son - two different needs, two different subjects). Do not split "
            "a single need into multiple entries just because it has several details attached to it."
        ),
    )
    need_is_specific: YesNo = Field(
        ...,
        description=(
            "'yes' if the need is specific enough to search for directly - e.g. names a "
            "scheme, a benefit that states what it is FOR (pension/scholarship/healthcare/...), "
            "or a clear purpose - "
            "EVEN IF you can't map it to one of the exact category/benefit_type labels above. "
            "'no' if it's too broad/vague to know what KIND of help they want "
            "(e.g. 'I need money', 'help chahiye' with no purpose stated).\n"
            "CRITICAL - judge the PURPOSE only, never the amount of detail. Stating WHO THEY ARE "
            "or WHAT THEY OWN is background context, NOT a purpose, and does not make a need "
            "specific no matter how much of it is given. Found live (2026-09-02): 'mai farmer hu "
            "mere paas 5 acre zameen hai mere liye koi yojna btao' was marked 'yes' - it is 'no'. "
            "Occupation and landholding say nothing about what help is wanted; a farmer can need a "
            "pension, equipment subsidy, irrigation infrastructure, a crop loan, training, "
            "insurance, or a scholarship for their child, and 'koi yojna btao' picks none of them. "
            "The test is a single question: after reading this message, do you know what the help "
            "is FOR? If the only way to answer is to guess among several unrelated kinds of "
            "benefit, answer 'no'. Same for 'mai student hu koi scheme batao', 'mai vidhwa hu kuch "
            "madad chahiye', 'mere paas dukan hai koi yojna hai kya' - all identity plus a bare "
            "request, all 'no'.\n"
            "Naming only how the money is DELIVERED is not a purpose either. 'Subsidy', 'loan', "
            "'grant', 'anudan', 'sahayata' are funding mechanisms - nearly every scheme is one of "
            "them, so they say nothing about what the help is for. 'Mujhe subsidy chahiye' or "
            "'loan chahiye', with no statement of what it is FOR, is 'no' (found live 2026-09-02: "
            "a farmer answered 'mujhe subsidy chahiye' and it was treated as specific, but "
            "equipment, irrigation, storage and orchard subsidies are all still in play). "
            "Contrast 'pension' and 'scholarship', which DO say what the help is for (old-age "
            "income, education) - those are 'yes'.\n"
            "Naming the SECTOR, TRADE or VENTURE is likewise not a purpose. 'Solar ka business "
            "start karna chahta hu', 'dairy ka kaam shuru karna hai', 'kheti ke liye madad "
            "chahiye' all name the field of activity while leaving the actual ask wide open - "
            "capital, equipment, training, premises, licensing help, an interest subsidy? - so "
            "they are 'no'. Found live (2026-09-02): 'mai solar ka business start karna chahta hu' "
            "was marked 'yes' purely because 'solar' is a concrete word, and the reply jumped "
            "straight to naming a scheme. Adding a specific noun to a vague request does not make "
            "the request specific - note that the SAME message with the noun removed ('mai apna "
            "business start karna chahta hu') is correctly judged 'no', and nothing about what the "
            "citizen needs actually differs between the two.\n"
            "The test that decides it: could you name the KIND of benefit they want - money for a "
            "named thing, equipment, training, a pension, treatment, housing, fees? If yes, 'yes'; "
            "if you would have to guess between several, 'no'. So 'padhai ke liye paisa' (money "
            "for studies), 'bimari ka kharcha' (medical costs) and 'khet ke yantra ke liye "
            "subsidy' (equipment) are all 'yes' - each names the benefit itself, not just the "
            "area of life it belongs to."
        ),
    )
    intent: Intent = Field(
        ...,
        description=(
            "What KIND of question this is, which decides which scheme fields get fetched:\n"
            "'how_to_apply' - asks about the APPLICATION PROCESS, even indirectly - not just "
            "'how do I apply', but also 'is there an online portal', 'where do I submit this', "
            "'online ya offline', 'kahan jama karu', asking for a form, a website, an office to "
            "visit. Fetches application+documents data.\n"
            "'documents_question' - asks specifically what documents/papers/certificates are "
            "needed. Fetches documents data only.\n"
            "'ask_detail' - asks about eligibility, benefits, amounts, who qualifies, or wants "
            "more general information about an already-identified scheme - anything that isn't "
            "specifically about the application process or documents.\n"
            "'find_scheme' - describes a need/purpose and wants a scheme found for it. Use this "
            "for 'is there anything else'/'koi aur scheme'/'iske alawa kuch aur' too - wanting "
            "OTHER options is still a find_scheme request; 'wants_alternatives' is a turn_type "
            "value below, never a valid value for THIS field.\n"
            "'chitchat' - small talk, thanks, greetings - not a scheme question at all."
        ),
    )
    turn_type: TurnType = Field(
        ...,
        description=(
            "EXACTLY one, mutually exclusive - do not reason about these as separate yes/no "
            "questions, pick the single best-fitting category. Applies IDENTICALLY whether the "
            "domain is a scheme or a JOB posting - the name says 'scheme' but every rule and "
            "example below applies the same way to a job already shown (found live 2026-08-19: "
            "this description used to give ONLY scheme-flavored examples, with no job example "
            "anywhere, and job followups got misclassified/stuck on a stale shown job across "
            "several later, unrelated questions more often as a result):\n"
            "'new_need' - the message describes a DIFFERENT need/topic than what's currently "
            "shown (even if phrased casually) - e.g. 'and if i want to start a business' after a "
            "scholarship was shown, or asking about a completely different job/post than the one "
            "just shown.\n"
            "'followup_same_scheme' - asks for more detail about the SAME scheme OR JOB already "
            "shown (documents, how to apply, eligibility/qualification, salary, deadline, a "
            "pronoun/implicit reference like 'this'/'iske') - including terse procedural questions "
            "with no topic of their own. Only pick this if the message is STILL plausibly about "
            "that same specific scheme/job - if the topic has clearly moved on (even a few turns "
            "back), that's 'new_need', not this, no matter how recently it was shown.\n"
            "'wants_alternatives' - explicitly asks for OTHER/additional/more schemes OR JOBS for "
            "the SAME underlying need already established (e.g. 'what other schemes am I eligible "
            "for', 'any other jobs available', 'aur konsi jobs hain') - NOT a message that itself "
            "describes a new need, which is 'new_need' regardless of how it's phrased."
        ),
    )
    detected_language: ReplyLanguage = Field(
        ...,
        description=(
            "Which register the CITIZEN'S CURRENT message is actually written in - the reply "
            "must match this exactly, so classify carefully:\n"
            "'hindi' - written in Devanagari script (देवनागरी) ONLY - never any other Indian "
            "script (Gurmukhi/Punjabi, Bengali, Gujarati, etc.) - this app supports exactly these "
            "three registers, nothing else. This is a SCRIPT rule, not a vocabulary one - classify "
            "'hindi' even if most of the individual WORDS are English loanwords, as long as they're "
            "spelled out in Devanagari. Found live (2026-08-26): 'मैं अभी इलेवंथ में हूँ ... क्या मुझे "
            "हायर एजुकेशन्स के लिए ... स्कीम्स हैं ... अवेलेबल' ('eleventh'/'higher educations'/"
            "'schemes'/'available', all spelled phonetically in Devanagari) is just as much 'hindi' "
            "as a message with zero English loanwords - do not let a loanword-heavy message read as "
            "'basically English, so reply in Hinglish/English register instead'.\n"
            "'hinglish' - Hindi/Rajasthani words spelled out in Roman/Latin letters (e.g. 'mujhe "
            "scheme chahiye', 'mera beta 12th mein hai') - this is the common register for rural "
            "Hindi speakers typing in Roman script. Classify as hinglish even if a few English "
            "words are mixed in, as long as the sentence is fundamentally Hindi grammar/vocabulary "
            "in Roman letters.\n"
            "'english' - genuine English vocabulary and grammar, not Romanized Hindi.\n"
            "Base this STRICTLY on the citizen's message in THIS turn ALONE, not on what language "
            "previous turns used - do not let the surrounding conversation's language bias this "
            "classification. Found live (2026-08-19): 'give me update regarding Lab assistant' - "
            "unambiguous English vocabulary and grammar with not one Hindi/Hinglish word in it - "
            "still got classified as 'hinglish', purely because the rest of that conversation had "
            "been in Hindi/Hinglish. A short message surrounded by a different-language "
            "conversation is judged ENTIRELY on its own words - if this exact message, with no "
            "context at all, would obviously read as English, classify it 'english', full stop, "
            "even if every other turn around it was Hindi/Hinglish."
        ),
    )
    followup_target: Optional[str] = Field(
        None,
        description=(
            "ONLY relevant when several schemes OR jobs were listed above as shown last turn AND "
            "turn_type='followup_same_scheme': copy the EXACT scheme_name (or job title, if jobs "
            "were shown) from that list that this message is asking about - by position ('pehli'/"
            "'first' -> the 1st one listed, 'dusri'/'second' -> the 2nd), by topic ('education "
            "wali' -> whichever is about education), or by name. null if only one item was shown "
            "(nothing to disambiguate), or if the message clearly refers to ALL of them together."
        ),
    )
    chitchat_reply: Optional[str] = Field(
        None,
        description=(
            "ONLY if intent='chitchat': write the actual reply to send back, right here - not a "
            "description of what to say, the literal reply text itself. Keep it brief and warm, "
            "in the same language/register as detected_language above, and mention you can help "
            "them find Rajasthan government schemes and jobs. Word it naturally and freshly - if "
            "the recent conversation above shows an earlier reply of yours, do not reuse its "
            "opening words or repeat the same stock acknowledgment. null for every other intent - "
            "do not write anything here unless intent is genuinely 'chitchat'."
        ),
    )
    is_answering_pending_question: Optional[bool] = Field(
        None,
        description=(
            "ONLY relevant when the context above tells you a question was left pending from your "
            "last reply - leave this null if it doesn't mention one. When it does, decide: true if "
            "THIS message is actually a direct answer to that specific question (a fact about "
            "themselves, a yes/no, a choice between options it offered) - false if this message is "
            "about something else entirely, even though a question was left pending - e.g. it names "
            "a different/specific scheme or job to ask about instead, asks to switch reply language "
            "or format, asks you to pause/wait, or is otherwise unrelated to what was asked. Do not "
            "default to true just because a question was pending - read what THIS message actually "
            "says and judge whether it answers that question or not."
        ),
    )
    stated_facts: list[str] = Field(
        default_factory=list,
        description=(
            "Any specific, checkable fact the citizen just confirmed or denied about their own "
            "eligibility situation THIS message - phrase it as a short, plain, self-contained "
            "statement worth remembering for the REST of the conversation, not just this reply "
            "(e.g. 'does not yet have an MSME/Udyam registration certificate', 'owns a tractor "
            "registered in their own name', 'has not claimed this type of subsidy before', 'business "
            "is still at the idea stage, not yet started'). This is for narrow, scheme-specific "
            "criteria that AREN'T one of the identity fields above (caste/gender/marital_status/"
            "residence/education/disability/occupation already have their own tracking) - things "
            "like registrations, ownership of specific assets, or prior scheme history. Only include "
            "NEW facts stated or confirmed this message - do not restate anything already known from "
            "earlier in the conversation. Empty list if nothing new and checkable was stated."
        ),
    )


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def normalize_node(state: ConvState):
    raw = state["raw_message"].strip()
    result = {
        # Cleared at the START of every turn, not just after use - if a
        # PREVIOUS turn set this (janaadhaar_handle_node redacting an ID/OTP
        # attempt), it must not linger and wrongly redact a later, unrelated
        # turn's conversation_history entry. janaadhaar_handle_node re-sets
        # this later in THIS SAME turn's run if it actually processes one.
        "redacted_raw_message": None,
        # Same reset reasoning as above - a mock OTP popup from an earlier
        # turn must not resurface on an unrelated later turn.
        "mock_sms_preview": None,
    }
    if not state.get("pending_janaadhaar_step"):
        # General, content-based safety net (2026-08-25, found live) - every
        # OTHER Jan Aadhaar redaction only fires once pending_janaadhaar_step
        # is already set, so a citizen who pastes their ID UNPROMPTED, in
        # the very message that first raises a citizen_status need, has it
        # sent straight to extract_profile_node/extract_need_node's LLM
        # prompts before the graph even knows a Jan Aadhaar flow is
        # relevant - see pii_utils.scrub_jan_aadhaar_ids' own docstring.
        # Overwrites raw_message itself, not just normalized_message -
        # generate_node/generate_jobs_node both read state["raw_message"]
        # directly ("Citizen's exact words this turn"), bypassing
        # normalized_message entirely.
        scrubbed = pii_utils.scrub_jan_aadhaar_ids(raw)
        if scrubbed != raw:
            raw = scrubbed
            result["raw_message"] = scrubbed
            result["redacted_raw_message"] = scrubbed
        # Cheap placeholder only - extract_need_node's detected_language
        # always overrides this with a real 3-way classification (hindi/
        # hinglish/english) before any reply-generating node reads
        # reply_lang. This regex can only tell Devanagari from
        # not-Devanagari, so it can't tell Hinglish from English - kept only
        # as a fallback value, in the unlikely case something reads
        # reply_lang before extract_need_node completes.
        # Skipped entirely while a Jan Aadhaar verification turn is pending
        # (route_after_normalize is about to send this straight to
        # janaadhaar_handle_node, bypassing extract_need_node) - found live
        # (2026-08-14): without this guard, a Hinglish-typed Jan Aadhaar ID
        # ("mera jan aadhaar id ... hai", no Devanagari characters) got this
        # crude heuristic's "english" guess instead, silently downgrading a
        # conversation already locked to Hinglish since this heuristic's
        # value was never subsequently corrected by extract_need_node on
        # these bypassed turns.
        result["reply_lang"] = "hindi" if DEVANAGARI_RE.search(raw) else "english"
    result["normalized_message"] = raw
    return result


def route_after_normalize(state: ConvState):
    """Intercepts every turn where a Jan Aadhaar verification sub-flow is
    already in progress (an ID, member choice, or OTP is expected next) and
    sends it straight to janaadhaar_handle_node, BYPASSING extract_profile/
    extract_need entirely - both are LLM calls, and this turn's raw_message
    IS itself a credential the citizen just typed. Every other turn proceeds
    through the normal extraction pipeline unchanged."""
    if state.get("pending_janaadhaar_step"):
        return "janaadhaar_handle"
    return "extract_profile"


def extract_profile_node(state: ConvState):
    profile = state.get("profile", {})
    # Recent turns, not just the current message in isolation - found live
    # (2026-08-11): "this is her first marriage" (mid-conversation about the
    # citizen's DAUGHTER's wedding) got wrongly attributed as the citizen's
    # OWN marital_status, purely because this node had no way to know the
    # conversation was currently about someone else. Verified directly: the
    # model correctly identifies whose fact a message states once it can
    # actually see what's being discussed - it wasn't a reasoning gap, it
    # was missing the same context extract_need_node already gets.
    history_block = _format_conversation_history(state.get("conversation_history"))
    existing_notes = state.get("conversation_notes") or []
    notes_block = (
        "Facts already noted this conversation (do not repeat these in new_notes):\n"
        + "\n".join(f"- {n}" for n in existing_notes) + "\n\n"
    ) if existing_notes else ""
    prompt = (
        history_block
        + (f"Known facts about this citizen so far: {profile}\n\n" if profile else "")
        + notes_block
        + f'Citizen\'s new message: "{state["normalized_message"]}"\n\n'
        + "Extract only NEW facts about the citizen THEMSELVES (the person sending this message) "
        "stated in THIS message. Use the recent conversation above to tell whether this message is "
        "about the citizen or about someone else (e.g. a family member) already being discussed - a "
        "message continuing a conversation about a family member's situation is still about THAT "
        "person, not the citizen, even if it doesn't re-name them.\n\n"
        "For EVERY field: fill it in ONLY if the citizen explicitly and literally stated that "
        "fact about themselves. Never fill in a field because of an association, stereotype, or "
        "plausible-sounding guess connected to something else in the message - not from a place "
        "name, not from an occupation, not from their age, not from a fact about a family member, "
        "not from the general tone or completeness of their self-introduction. A detail is only "
        "evidence for a field if it is THAT field, stated directly - anything else, no matter how "
        "commonly correlated in the real world, is not evidence and must not be used. "
        "This holds even when the citizen states several other facts in the same message - stating "
        "more facts about themselves is not itself a reason to fill in additional, unstated fields. "
        "For example: naming a family member's need or trait (a scholarship 'for my beti/beta') is "
        "not a fact about the citizen's own gender, marital status, or education; naming a place "
        "they live in (a state or city name) is not, by itself, evidence of rural or urban - only "
        "an explicit description of where they live is; naming their occupation is not evidence of "
        "their education level or income - only an explicit statement about schooling, a degree, or "
        "an income figure is. Generalize this same standard to every field, not just these examples. "
        "When genuinely uncertain whether something counts as an explicit statement about the "
        "citizen themselves, leave the field null rather than guess. "
        "DENYING one specific value is NOT the same as confirming any value, including a different "
        "one: if the citizen says they are NOT something (e.g. 'I am not transgender', 'main "
        "transgender nahi hoon'), that rules out ONLY the denied value - it does not mean the field "
        "should be filled with a different guessed value (e.g. do not fill gender with 'female' "
        "just because 'transgender' was denied), and it absolutely does NOT mean the denied value "
        "itself should be filled in either (never output 'transgender' here because the citizen "
        "said they are NOT transgender - read the negation carefully, especially across 'nahi ... "
        "nahi hoon'-style Hindi negation). A denial with no separate positive statement leaves the "
        "field null. "
        "Do not restate facts already known above unless this message corrects them. "
        "Most messages state no new profile facts at all - that's fine, leave those null.\n\n"
        "Separately, in new_notes, record any OTHER durable facts from THIS message per that "
        "field's own description - facts with no profile field of their own (family members, "
        "documents, prior applications, assets, job preferences, direct answers to a question "
        "your last reply asked). The same explicitly-stated-only standard applies there too."
    )
    result = _invoke_structured(get_llm(), ProfileExtraction, prompt, label="extract_profile")

    # Evidence-grounding backstop for `residence` (2026-08-10): two rounds of
    # prompt-only guardrails failed to stop the model inferring rural/urban
    # from a place NAME's stereotype (e.g. "Rajasthan" -> rural) when another
    # fact was also stated - confirmed live, 2/2 reproducing, on both prompt
    # attempts. Rather than enumerate the words that count as real evidence
    # (a keyword list that can never keep pace with real phrasing diversity),
    # require the model to cite its own justification and verify that
    # citation is real: if there's no evidence, or the "evidence" isn't
    # actually present in the message, the classification is unfounded.
    evidence = (result.residence_evidence or "").strip().lower()
    if result.residence is not None and (not evidence or evidence not in state["normalized_message"].lower()):
        result.residence = None

    # Same evidence-grounding check for gender (2026-08-11): "rehne wala hu"
    # (masculine) was inconsistently NOT recognized as evidence while "rehne
    # wali hu" (feminine) reliably was, even after a prompt-only fix. Forcing
    # a cited quote doesn't stop hallucination here (the earlier failure was
    # under-recognition, not over-confident guessing) but requires the model
    # to actively locate real evidence before committing to a value, and a
    # claim with no real backing text is dropped either way.
    gender_evidence = (result.gender_evidence or "").strip().lower()
    if result.gender is not None and (not gender_evidence or gender_evidence not in state["normalized_message"].lower()):
        result.gender = None

    merged_profile = dict(profile)
    for field in ("caste", "gender", "marital_status", "age", "family_income", "residence", "education_level", "disability"):
        value = getattr(result, field)
        if value is not None:
            merged_profile[field] = value

    # Union, not overwrite - unlike the scalar fields above, occupation isn't
    # exclusive (a citizen can be a farmer AND run a business at the same
    # time), so a later message naming a different occupation must ADD to
    # what's already confirmed, never replace it. Found live (2026-08-17):
    # occupation confirmation used to rely on a keyword match against
    # whatever need_text happened to be in play THIS turn (see _identity_gap)
    # instead of a persisted fact - it only "worked" by coincidence when a
    # scheme name itself contained a matching keyword (e.g. "Krishi Yantra"),
    # and broke the moment the conversation moved to a differently-named
    # scheme (Walk In Tunnel) despite the citizen's farmer status having been
    # stated turns earlier.
    if result.occupation_groups:
        merged_profile["occupation_groups"] = sorted(
            set(profile.get("occupation_groups") or []) | set(result.occupation_groups)
        )

    # Append-only free-form memory (see ConvState.conversation_notes) - this
    # is what lets an out-of-schema fact ("son is in class 10", "no
    # disability certificate") outlive the rolling conversation_history
    # window instead of being re-asked.
    return {"profile": merged_profile, "conversation_notes": _merge_notes(existing_notes, result.new_notes)}


def extract_need_node(state: ConvState):
    shown_scheme_name = state.get("shown_scheme_name")
    shown_schemes = state.get("shown_schemes") or []
    shown_jobs = state.get("shown_jobs") or []
    last_reply = state.get("last_reply")
    # Captured now, before pending_need_domain gets reset below (same
    # lifecycle reason pending_need_text is captured into a local here too) -
    # route_after_extract needs to know which domain THIS resolved
    # clarification belonged to, and by the time it runs, state's own
    # pending_need_domain would already be the freshly-reset None.
    pending_need_domain_before = state.get("pending_need_domain")
    pending_need_text = state.get("pending_need_text")

    context_lines = []
    history = state.get("conversation_history")
    if history:
        # Full recent turns, not just the last reply - lets the model resolve
        # WHO a terse pronoun-only followup ("no she is not disable") refers
        # to by actually reading who was being discussed, the same way a
        # human would, instead of us pre-computing and threading a separate
        # "current subject" field through state. Found live (2026-08-10): a
        # followup about a family member (a citizen's wife) silently reverted
        # to being about the citizen themselves once the message no longer
        # named "wife" explicitly, because this node previously had no
        # access to the turn where "wife" was first established.
        context_lines.append(_format_conversation_history(history).rstrip())
    notes = state.get("conversation_notes") or []
    if notes:
        # Durable facts that may predate the rolling history window above -
        # lets terse followups resolve against facts stated many turns ago
        # (e.g. a job qualification or a family member's situation).
        context_lines.append(
            "Durable facts noted earlier this conversation:\n" + "\n".join(f"- {n}" for n in notes)
        )
    if len(shown_schemes) > 1:
        # Multiple schemes shown at once (a multi-need answer, or several
        # options) - give the model the actual list so it can resolve which
        # one a followup means, instead of us guessing positionally in code.
        # Found live (2026-08-10): "pehli scheme"/"dusri scheme" (first/
        # second scheme) followups had nothing to resolve against and got
        # answered against a stale single shown_scheme_id from an unrelated
        # earlier need.
        options = "; ".join(
            f'{i+1}. "{s["scheme_name"]}" (about: {s.get("need_text", "")})'
            for i, s in enumerate(shown_schemes)
        )
        context_lines.append(f"Multiple schemes were shown last turn, in this order: {options}")
    elif shown_scheme_name:
        context_lines.append(f"Scheme currently in focus (most recently shown): {shown_scheme_name}")
    if len(shown_jobs) > 1:
        job_options = "; ".join(f'{i+1}. "{j["title"]}" ({j.get("department", "")})' for i, j in enumerate(shown_jobs))
        context_lines.append(f"Multiple job postings were shown last turn, in this order: {job_options}")
    elif shown_jobs:
        context_lines.append(f"Job posting currently in focus (most recently shown): {shown_jobs[0]['title']}")
    if last_reply:
        context_lines.append(f"Your last reply to them was: {last_reply}")
    if pending_need_text:
        context_lines.append(
            f'They were previously asking about: "{pending_need_text}" - if this message is just a '
            "short answer to your last reply (yes/no/a fact about themselves) rather than a full "
            "restatement, treat need_text as still being about that."
        )
    locked_lang = state.get("locked_reply_lang")
    if locked_lang:
        context_lines.append(
            f"The citizen has picked '{locked_lang}' as their language for this whole conversation - "
            f"if you fill in chitchat_reply below, write it in '{locked_lang}' specifically, "
            "regardless of what detected_language would otherwise classify this message as."
        )
    context = "\n".join(context_lines)

    prompt = (
        f"{context}\n\n"
        f'Citizen\'s message: "{state["normalized_message"]}"\n\n'
        "Classify what they want and how clear it is."
    )

    result = _invoke_structured(get_llm(), NeedExtraction, prompt, label="extract_need")

    needs = []
    for item in result.needs:
        needs.append({
            "need_text": item.need_text,
            "need_category": _normalize_to_known(item.need_category, KNOWN_CATEGORIES),
            "need_benefit_type": _normalize_to_known(item.need_benefit_type, KNOWN_BENEFIT_TYPES),
            "domain": item.domain,
            "job_location": item.job_location,
            "job_qualification": item.job_qualification,
            "job_min_salary": item.job_min_salary,
            "about": item.about,
            "subject_relation": item.subject_relation,
            "subject_facts": {
                k: v for k, v in {
                    "caste": item.subject_caste,
                    "gender": item.subject_gender,
                    "education_level": item.subject_education_level,
                    "disability": item.subject_disability,
                    "age": item.subject_age,
                    "marital_status": item.subject_marital_status,
                    "family_income": item.subject_family_income,
                    # Same evidence-grounding rule as the citizen's own residence
                    # (2026-08-10 fix): only trust it if the cited evidence is a
                    # real substring of the message, not just a place name.
                    "residence": (
                        item.subject_residence
                        if item.subject_residence is not None
                        and (item.subject_residence_evidence or "").strip().lower() in state["normalized_message"].lower()
                        else None
                    ),
                }.items() if v is not None
            },
        })
    if not needs:  # defensive - the model is expected to always return >=1
        # Found live (2026-08-14): a fully context-free, very generic message
        # ("what is my application status", first turn, nothing else stated)
        # can come back with an empty needs list from the model. domain
        # defaults to "scheme" here - matching NeedItem.domain's own "when
        # genuinely unclear, default to scheme" rule - so route_after_extract's
        # domain checks (job/citizen_status) have a defined value to read
        # instead of silently falling through on a missing key.
        needs = [{"need_text": state["normalized_message"], "need_category": None, "need_benefit_type": None,
                  "domain": "scheme", "about": "self", "subject_relation": None, "subject_facts": {}}]
    primary = needs[0]

    need_is_specific = result.need_is_specific == "yes"

    intent = result.intent
    if intent == "chitchat" and (primary["need_category"] or primary["need_benefit_type"] or need_is_specific):
        # the model contradicted itself - it extracted a real need but also
        # called this chitchat. A stated goal ("I want to start a business")
        # isn't idle chat just because it lacks a question mark; trust the
        # extracted need over the intent label when they conflict.
        intent = "find_scheme"

    result_dict = {
        "needs": needs,
        "need_text": primary["need_text"],
        "need_category": primary["need_category"],
        "need_benefit_type": primary["need_benefit_type"],
        "need_is_specific": need_is_specific,
        "intent": intent,
        "turn_type": result.turn_type,
        # A citizen-picked language (via the frontend's language picker)
        # always wins over this turn's own detection - see ConvState.
        # locked_reply_lang's docstring for why (voice transcription
        # flip-flopping between Devanagari and Romanized Hindi).
        "reply_lang": state.get("locked_reply_lang") or result.detected_language,
        "followup_target": result.followup_target,
        "pending_need_text": None,  # consumed - don't let it linger past the turn it was needed for
        "pending_need_domain": None,  # consumed alongside pending_need_text above - same lifecycle
        # was the PREVIOUS turn a clarifying question (clarify/clarify_identity/
        # generate), AND is THIS message actually answering it? route_after_extract
        # uses this to force a fresh Stage A run regardless of what turn_type
        # says. Found live (2026-08-05): "education"/"undergraduate"/"she is
        # 12th pass" - answers to the bot's own clarifying questions - kept
        # getting misrouted to stage_b_followup against a stale shown_scheme
        # from many turns earlier (a different topic entirely), producing
        # answers about the wrong scheme and a dead-end loop the citizen
        # couldn't escape even by explicitly asking for a different scheme -
        # hence forcing stage_a whenever a question was left pending.
        # But a question being pending doesn't mean THIS message answers it -
        # found live (2026-08-17): "hindi me likho" and (separately) naming a
        # specific already-shown scheme by name both got forced through this
        # same override and derailed into a fresh, irrelevant search, because
        # it used to fire on bool(pending_need_text) alone with no check on
        # what the new message actually says. is_answering_pending_question is
        # the model's own direct judgment call on that, made with the pending
        # question already in its context (see context_lines above) - default
        # to the old conservative behavior (true) unless it explicitly says
        # false, since a false negative here reproduces the 2026-08-05 bug.
        "was_answering_clarification": (
            bool(pending_need_text) and result.is_answering_pending_question is not False
        ),
        # Which domain that resolved clarification actually belongs to -
        # route_after_extract reads THIS (not pending_need_domain, which is
        # already reset to None above) to decide "stage_a" vs "stage_a_jobs"
        # when was_answering_clarification fires. Found live (2026-08-18):
        # "mere liye koi job btao" -> clarify_jobs asks about qualification
        # -> "10th pass hu" (answering it) got forced into a fresh SCHEME
        # search instead of continuing the job search, because the override
        # always routed to "stage_a" with no awareness a JOB clarification
        # could be the one pending - surfaced a completely unrelated
        # "Leather Training Scheme" with no connection to the job request.
        "clarification_domain": pending_need_domain_before,
    }
    if result.stated_facts:
        # Additive, never reset - facts about the citizen/their situation
        # (registrations, asset ownership, prior scheme history), not tied
        # to whichever need is currently being discussed, so they must
        # survive a topic switch the same way `profile` does. Feeds the SAME
        # conversation_notes list as extract_profile_node's new_notes (this
        # node runs after it, so state already carries this turn's profile-
        # channel notes) - one durable store, two capture channels, since
        # this call sees the need/clarify-answer context the profile call
        # doesn't. Added 2026-08-18: narrow scheme-specific criteria (MSME
        # registration, tractor ownership, ...) previously lived only in
        # conversation_history's rolling window, so clarify_identity_node/
        # generate_node re-asked them once the window scrolled past the turn
        # where they were first confirmed.
        result_dict["conversation_notes"] = _merge_notes(state.get("conversation_notes"), result.stated_facts)
    if not result_dict["was_answering_clarification"]:
        # No clarification is being continued into THIS turn - either
        # nothing was pending at all, or something WAS pending but this
        # message answers something else / changes topic instead - so
        # there's no unresolved clarify streak to cap here, reset the
        # counter. Was gated on bare `not pending_need_text` (2026-08-25,
        # found live) - that only caught the first case, not the second: a
        # citizen who ignores a pending clarifying question and states a
        # brand-new need instead correctly got was_answering_clarification=
        # False (routing behaved right), but clarify_attempts/
        # generate_clarify_rounds still carried over unearned from the
        # abandoned exchange - e.g. an abandoned SCHEME clarification
        # silently used up the JOB domain's own single clarify_jobs
        # attempt for an unrelated later job search, skipping its narrowing
        # question entirely. Left untouched (not returned) only while a
        # clarification IS actually being continued, so the streak still
        # survives across the turns route_after_stage_a needs to count it
        # over.
        result_dict["clarify_attempts"] = 0
        result_dict["generate_clarify_rounds"] = 0

    # The reply itself was already composed as part of this SAME call
    # (chitchat_reply) - route_after_extract sends this straight to
    # finalize instead of a separate chitchat_node call when this is set,
    # cutting chitchat turns from 3 LLM calls down to 2 (extract_profile +
    # this call). Falls back to the old separate chitchat_node call (still
    # wired into the graph) if the model left chitchat_reply empty despite
    # intent='chitchat' - never silently produce an empty reply.
    if intent == "chitchat" and result.chitchat_reply:
        result_dict["last_reply"] = result.chitchat_reply
    return result_dict


def route_after_extract(state: ConvState):
    # was_answering_clarification already factors in the model's own
    # is_answering_pending_question judgment (see extract_need_node) - no
    # need to special-case intent/turn_type/followup_target here as well.
    # clarification_domain decides WHICH search this resumes into - found
    # live (2026-08-18): this used to unconditionally return "stage_a" (the
    # SCHEME search) regardless of domain, so answering a JOB clarifying
    # question (clarify_jobs) got forced into a fresh, unrelated scheme
    # search instead of continuing the job search.
    if state.get("was_answering_clarification"):
        return "stage_a_jobs" if state.get("clarification_domain") == "job" else "stage_a"
    turn_type = state.get("turn_type")
    primary_need = (state.get("needs") or [None])[0]
    # A citizen-status question ("kya meri pension aayegi") can only be
    # answered by looking up THIS citizen's own record - kicks off the Jan
    # Aadhaar OTP-consent sub-flow instead of a scheme/job search. This turn's
    # own message (the status question itself) has no identifier in it, so
    # it's safe that it already went through extract_profile/extract_need
    # above - only the FOLLOWING turns (ID/OTP) get intercepted earlier, by
    # route_after_normalize.
    if primary_need and primary_need.get("domain") == "citizen_status":
        return "janaadhaar_handle"
    # Job vacancies are a separate domain from welfare schemes (no identity
    # eligibility gating, structured qualification/location/deadline data
    # instead) - routed to their own parallel subgraph rather than through
    # Stage A's scheme-shortlist/clarify machinery, which doesn't apply here.
    if primary_need and primary_need.get("domain") == "job":
        if turn_type == "followup_same_scheme" and state.get("shown_job_id"):
            return "stage_b_followup_jobs"
        return "stage_a_jobs"  # covers both a fresh job search and "show other jobs" in v1
    if turn_type == "wants_alternatives":
        shortlist = state.get("shortlist") or []
        already_shown = set(state.get("already_shown_scheme_ids") or [])
        remaining = [s for s in shortlist if s["scheme_id"] not in already_shown]
        if remaining:
            return "show_more"
        return "stage_a"  # nothing left to show from - fresh search instead of erroring
    if turn_type == "followup_same_scheme" and state.get("shown_scheme_id"):
        return "stage_b_followup"
    if state.get("intent") == "chitchat":
        # extract_need_node already composed the reply in the same call
        # (chitchat_reply) - skip the separate chitchat_node call entirely
        # and go straight to finalize. Only falls through to the old
        # chitchat_node path if that field came back empty.
        return "finalize" if state.get("last_reply") else "chitchat"
    return "stage_a"


def stage_a_node(state: ConvState):
    profile = state.get("profile", {})
    needs = state.get("needs") or [{
        "need_text": state["need_text"], "need_category": state.get("need_category"),
        "need_benefit_type": state.get("need_benefit_type"), "about": "self", "subject_facts": {},
    }]
    svc = get_rag_service()

    shortlists = []
    # (citizen, alt_citizen) per need, in the same order as `needs` - kept so
    # the re-pin check at the end of this function can re-run the eligibility
    # verdict for the PRIMARY need without rebuilding its context, and
    # without accidentally using the last loop iteration's values on a
    # multi-need turn.
    citizen_by_need = []
    for need in needs:
        facts = _subject_facts_for_need(need, profile)
        citizen = CitizenContext(
            caste=facts.get("caste"),
            gender=facts.get("gender"),
            marital_status=facts.get("marital_status"),
            age=facts.get("age"),
            family_income=facts.get("family_income"),
            residence=facts.get("residence"),
            education_level=facts.get("education_level"),
            disability=facts.get("disability"),
            need_category=need.get("need_category"),
            need_benefit_type=need.get("need_benefit_type"),
        )
        # For a family-member need, also check the citizen's OWN profile as an
        # alternate candidate identity - some schemes "for" a family member
        # actually gate eligibility on the APPLICANT (e.g. a widow's-
        # daughter's-marriage scheme requires the widow/citizen's own marital
        # status, not the daughter's). A scheme should only be excluded if
        # NEITHER identity fits (see RagService.shortlist's alt_citizen doc).
        alt_citizen = None
        if need.get("about") == "family_member":
            alt_citizen = CitizenContext(
                caste=profile.get("caste"),
                gender=profile.get("gender"),
                marital_status=profile.get("marital_status"),
                age=profile.get("age"),
                family_income=profile.get("family_income"),
                residence=profile.get("residence"),
                education_level=profile.get("education_level"),
                disability=profile.get("disability"),
                need_category=need.get("need_category"),
                need_benefit_type=need.get("need_benefit_type"),
            )
        citizen_by_need.append((citizen, alt_citizen))
        trace = {}
        results = svc.shortlist(need["need_text"], citizen, alt_citizen=alt_citizen, top_n=10, trace=trace)
        # plain dicts, not ScoredScheme dataclass instances - LangGraph's
        # checkpointer only reliably (de)serializes plain/registered types.
        shortlist = [
            {"scheme_id": s.scheme_id, "scheme_name": s.scheme_name, "lang": s.lang, "score": s.score, "metadata": s.metadata}
            for s in results
        ]
        # Which scheme_id(s), if any, this need's own RagService.shortlist
        # call identified as a genuine, unambiguous NAME lookup (see
        # RagService.shortlist's name_boost_applied/name_matches trace
        # fields) - carried through to stage_b_node so it can narrow
        # focused_scheme_ids for a first-time "am I eligible for scheme X"
        # question, not just a followup on an already-shown scheme (2026-08-25,
        # found live - see stage_b_node's own comment on focused_scheme_ids).
        named_scheme_ids = list(trace["name_matches"].keys()) if trace.get("name_boost_applied") else []
        shortlists.append({
            "need": need, "shortlist": shortlist, "candidate_count": trace.get("survivors", len(results)),
            "named_scheme_ids": named_scheme_ids,
        })

    # Re-insert any scheme a clarifying question was asked ABOUT that this
    # turn's freshly re-ranked search dropped (see ConvState.
    # pending_clarify_scheme_ids for the live incident). Scoped to the turn
    # the citizen is actually ANSWERING that question - on any other turn a
    # stale pin would wrongly resurrect an old scheme. Pinned entries go to
    # the FRONT: the citizen just supplied the fact this scheme was waiting
    # on, so it is the most relevant candidate there is right now, and
    # stage_b_node only fetches the top 3.
    if state.get("was_answering_clarification"):
        pinned_ids = state.get("pending_clarify_scheme_ids") or []
        if pinned_ids:
            primary_shortlist = shortlists[0]["shortlist"]
            present = {s["scheme_id"] for s in primary_shortlist}
            # The facts used for THIS turn's primary need - the citizen's
            # answer is already folded into them, which is the whole point.
            primary_citizen, primary_alt = citizen_by_need[0]
            restored = []
            for scheme_id in pinned_ids:
                if scheme_id in present:
                    continue
                meta = svc._meta_by_scheme_id.get(scheme_id)
                if not meta:
                    continue
                # Only re-pin a scheme the citizen's answer did NOT rule out.
                # Without this the pin would resurrect a scheme the answer
                # just disqualified (asked "SC/ST/OBC?", answered "general",
                # scheme is SC-only) and hand it back at rank 1, bypassing
                # the eligibility filter shortlist() had correctly applied -
                # turning a fix for "the right scheme vanished" into a much
                # worse "a disqualified scheme came back top of the list".
                # An answer that excludes the scheme is a RESOLVED question,
                # and the scheme dropping out is then the correct outcome.
                fits = eligibility_verdict(meta, primary_citizen)[0] or (
                    primary_alt is not None and eligibility_verdict(meta, primary_alt)[0]
                )
                if not fits:
                    continue
                restored.append({
                    "scheme_id": scheme_id,
                    "scheme_name": meta.get("scheme_name", ""),
                    "lang": meta.get("lang", "en"),
                    "score": 0.0,
                    "metadata": meta,
                })
            if restored:
                logger.info(
                    "(stage_a) re-pinned %d scheme(s) a clarifying question was about, dropped by this turn's re-ranked search: %s",
                    len(restored), [s["scheme_name"] for s in restored],
                )
                shortlists[0]["shortlist"] = restored + primary_shortlist

    primary = shortlists[0]
    return {
        "shortlists": shortlists,
        "shortlist": primary["shortlist"],
        "candidate_count": primary["candidate_count"],
    }


def route_after_stage_a(state: ConvState):
    shortlist = state.get("shortlist") or []
    if not shortlist:
        return "no_match"
    # Never clarify twice in a row for the same unresolved need - found live
    # (2026-08-10): a citizen who repeated the same vague answer twice got
    # the identical clarifying question back, verbatim, with no way out.
    # After one unanswered attempt, fall back to presenting the best
    # available candidates rather than asking again. This guard is scoped to
    # ONLY the vague-need branch below, not the identity-gap check further
    # down - found live (2026-08-24): once ANY clarifying question (of
    # either kind) had been asked once for a need, this used to skip the
    # identity-gap check FOREVER after, even when a brand-new, previously-
    # irrelevant gap surfaced later (e.g. confirming caste=sc narrows the
    # shortlist to a scheme with its own unconfirmed education_level gap).
    # That silently dumped straight to stage_b/generate_node instead of
    # asking, which then had to name a scheme it shouldn't have, or narrate
    # ruling schemes out in free text instead of asking a direct question.
    # The identity-gap check doesn't share this 1-attempt guard - it's
    # self-terminating in the NORMAL case (_first_identity_gap only ever
    # returns a gap that's genuinely still unconfirmed, so once the citizen
    # actually answers it, it stops appearing on its own), but still gets
    # its OWN, more generous cap (MAX_IDENTITY_CLARIFY_ATTEMPTS) for the
    # edge case where the citizen never directly answers it at all.
    if state.get("clarify_attempts", 0) < MAX_NEED_CLARIFY_ATTEMPTS:
        # need_is_specific is the model's own direct judgment of clarity - NOT
        # inferred from whether need_category/need_benefit_type happened to
        # populate (those exist only for the rank boost and are unreliable
        # proxies for "was the need actually clear").
        if not state.get("need_is_specific", True) and state.get("candidate_count", 0) >= CLARIFY_THRESHOLD:
            return "clarify"
    if state.get("clarify_attempts", 0) < MAX_IDENTITY_CLARIFY_ATTEMPTS:
        primary_need = (state.get("needs") or [None])[0]
        facts = _subject_facts_for_need(primary_need, state.get("profile", {}))
        need_text = primary_need["need_text"] if primary_need else state.get("need_text", "")
        _, gaps = _first_identity_gap(shortlist, facts, need_text)
        if gaps:
            return "clarify_identity"
    return "stage_b"


def route_after_stage_a_jobs(state: ConvState):
    """Jobs never had a clarify step at all (v1 gap - see stage_a_jobs_node's
    call site in route_after_extract) - a fully generic "mujhe job ke liye
    kuch btao" just ran a bare fuzzy search and dumped whatever came back,
    with no chance to ask what kind of job/where/what qualification first.

    NOT keyed on need_is_specific (tried first, found live 2026-08-18 to
    never actually fire): that field asks "is the general PURPOSE clear",
    and for jobs the purpose is nearly always clear ("I want a job" already
    states it) even when there's nothing to narrow WHICH job.

    NOT keyed on "did they state ANY of location/qualification/salary"
    either (tried next, found live 2026-08-20 to still under-ask): stating a
    qualification ("ITI aur BCom kar rakha hai") legitimately filters the
    search, but doesn't narrow it down to ONE job - "koi job batao" with a
    qualification still surfaced 3 different postings (2 different RAS
    locations + a Junior Assistant posting) with no chance to say which one
    actually matters to the citizen.

    A LOCATION alone used to be treated as sufficient on its own to skip
    straight to generate_jobs, regardless of how many distinct titles
    remained ("Jaipur mein job chahiye" was judged a genuine, deliberate
    preference worth honoring immediately). Reverted 2026-09-02, found live:
    "Jodhpur ka rehne wala hu, 12th pass kiya hai" gave a location AND a
    qualification, but never a job TYPE - and the location-only shortcut
    skipped the distinct-titles check entirely, dumping four completely
    unrelated postings (LDC, two different Lab Assistant listings, Junior
    Assistant) in one reply with no chance to say which kind of role
    actually mattered. Location narrows WHICH CITY's jobs are even in the
    shortlist (that filtering already happens in stage_a_jobs_node's own
    svc.search call) - it says nothing about WHICH TYPE of job within that
    city, which is a fully independent axis. Treating "a location is known"
    as license to also skip the type check was conflating two different
    questions.

    So the actual signal that matters, with no location-based exception:
    does the search still span MORE THAN ONE DISTINCT posting title? If so,
    ask which type before presenting a list, the same "narrow to one, don't
    dump a menu" discipline the scheme side already follows - regardless of
    whether location happens to be known too. (Trade-off, decided directly
    with the user: a bare "Jaipur mein job chahiye" with no type stated will
    now ALSO ask which type of post before showing anything, even though
    that specific phrasing was the original motivation for the shortcut this
    replaces - the user chose consistency over skipping that one round.)"""
    if state.get("clarify_attempts", 0) >= MAX_JOBS_CLARIFY_ATTEMPTS:
        return "generate_jobs"
    if state.get("job_title_named"):
        return "generate_jobs"
    shortlist = state.get("job_shortlist") or []
    distinct_titles = {j["title"] for j in shortlist}
    if len(distinct_titles) > 1:
        return "clarify_jobs"
    return "generate_jobs"


def clarify_jobs_node(state: ConvState):
    """Same shape as clarify_node, for the jobs domain - one short narrowing
    question (kind of job/location/qualification) instead of dumping a
    fuzzy-matched search result for a need that was never actually specific.

    Now reachable even when location/qualification ARE already known (see
    route_after_stage_a_jobs' 2026-09-02 update) - so, unlike before, this
    node can no longer assume nothing has been stated. Told explicitly what's
    already confirmed (known_block below), the same way generate_jobs_node
    already is, so it asks about the one axis that's actually still open
    (job type) instead of re-asking for a location or qualification the
    citizen already gave."""
    lang_instruction = _lang_instruction(state.get("reply_lang"))
    primary_need = (state.get("needs") or [None])[0] or {}
    known_bits = [
        f"{label}: {primary_need[field]}"
        for field, label in (("job_qualification", "qualification"), ("job_location", "preferred location"), ("job_min_salary", "minimum salary wanted"))
        if primary_need.get(field)
    ]
    known_block = (
        f"What's already confirmed about the citizen: {'; '.join(known_bits)} - do NOT ask about "
        "any of these again.\n" if known_bits
        else "Nothing about location, qualification, or specific post has been stated yet.\n"
    )
    prompt = (
        f"{_format_conversation_history(state.get('conversation_history'))}"
        f"{_format_profile(state.get('profile'))}"
        f"{_format_notes(state.get('conversation_notes'))}"
        f'The citizen asked about government jobs: "{state.get("need_text", "")}"\n'
        f"{known_block}"
        "This is still too broad to answer with one clear list - more than one genuinely different "
        "kind of posting currently matches.\n\n"
        "IMPORTANT: this dataset contains ONLY Rajasthan government job postings - there is no "
        "private-sector listing in it at all, in any field or company. Never ask whether they want "
        "a government or private-company job, or anything that presumes private options might "
        "exist to choose between - found live (2026-09-02): asked exactly that, the citizen said "
        "private, and the reply then had to walk it back a full turn later ('private IT jobs ka "
        "data abhi mere paas nahi hai'), wasting a turn that could have gone straight to the "
        "actual government postings. The only axes this dataset can actually be narrowed on are: "
        "which POST/title, which LOCATION/department, or the citizen's own QUALIFICATION.\n\n"
        f"Ask ONE short, simple clarifying question {lang_instruction} about whichever of those "
        "three axes is NOT already confirmed above (most often, at this point, which specific "
        "TYPE/POST they're after, since location and qualification are frequently already known) - "
        "it must be about something genuinely still open, never something already answered. Do not "
        "list job titles. One sentence only."
    )
    reply = _invoke_reply(get_reply_llm(), [("system", PERSONA), ("human", prompt)], state.get("reply_lang"), label="clarify_jobs")
    return {
        "last_reply": reply,
        "pending_need_text": state.get("need_text"),
        "pending_need_domain": "job",
        "clarify_attempts": state.get("clarify_attempts", 0) + 1,
    }


def clarify_node(state: ConvState):
    shortlist = state.get("shortlist") or []
    # Sensitive-gap-free-first, same lever as _prioritize_gap_free's own
    # docstring (found live 2026-08-11 for "apna business start karna hai"
    # leading with "are you disabled?"): a narrow/rare-population-gated
    # scheme shouldn't dominate the OPTIONS shown here either, not just the
    # schemes stage_b eventually explains. Found live again (2026-08-24) for
    # "mujhe scholarship chahiye": this node builds options_desc from the
    # raw, unfiltered shortlist - generate_node's identity-gap protection
    # never applies here at all, since a vague/broad need routes to THIS
    # node instead (route_after_stage_a's need_is_specific+CLARIFY_THRESHOLD
    # branch), before generate_node or _first_identity_gap ever run.
    #
    # _prioritize_gap_free (plain gap-free vs gapped) was tried here first
    # and had NO effect: every one of the top 10 shortlisted scholarships
    # had at least an education_level gap (school vs higher-ed was never
    # stated), so its gap-free bucket was empty and the reordering was a
    # no-op - the disability-gated scheme was still sitting in the options
    # list at its original rank, and got picked as the "cleanest" narrowing
    # axis anyway. _prioritize_sensitive_gap_free only demotes a SENSITIVE
    # gap (caste/disability/gender/marital-status - the same list
    # generate_node's own prompt calls out) so a scheme gapped only on a
    # broad fact like education_level still counts as clean here.
    primary_need = (state.get("needs") or [None])[0]
    facts = _subject_facts_for_need(primary_need, state.get("profile", {}))
    prioritized = _prioritize_sensitive_gap_free(shortlist, facts, state.get("need_text", ""), 8)
    # Discovery cards, not the bare (scheme_name, category) pair used before
    # 2026-09-02 - found live: "kheti ke liye sarkari madad chahiye" asked
    # about land size + irrigation source, despite the prompt below already
    # saying to prefer a "purpose" question first. Root cause was the data
    # this node judges purpose FROM, not the instruction: metadata['category']
    # is basically a constant across this whole domain (every single farming
    # scheme in the corpus - land-fencing, irrigation-digging, AND a genuinely
    # different farmer PENSION scheme - is tagged identically "Agriculture,
    # Rural & Environment"), so options_desc showed the model N options that
    # all displayed the same category, with nothing distinguishing "this one
    # is a pension" from "this one needs 1.5 hectares of irrigated land" -
    # purpose diversity that genuinely exists in the shortlist (the pension
    # scheme was very likely sitting right there among the 8) was invisible
    # in what got shown, so it fell back to whatever operational criterion
    # dominated the scheme NAMES instead. get_discovery_card already carries
    # benefit type + a real one-line description per scheme (per its own
    # docstring) - generate_node has used it for exactly this reason since
    # before this fix; clarify_node never had it wired in at all.
    svc = get_rag_service()
    lang = _source_lang(state.get("reply_lang"))
    options_desc = "\n".join(
        svc.get_discovery_card(s["scheme_id"], lang) or f"{s['scheme_name']} ({s['metadata'].get('category', '')})"
        for s in prioritized
    )
    lang_instruction = _lang_instruction(state.get("reply_lang"))
    # History/profile/notes context (2026-08-18): this node used to see NONE
    # of it - only the distilled need_text - so it could ask a narrowing
    # question the citizen had already answered a few turns earlier, which
    # read as the bot not listening.
    prompt = (
        f"{_format_conversation_history(state.get('conversation_history'))}"
        f"{_format_profile(state.get('profile'))}"
        f"{_format_notes(state.get('conversation_notes'))}"
        f'The citizen asked: "{state["need_text"]}"\n'
        f"This is too broad to answer directly - {state.get('candidate_count')} schemes could match, "
        f"including these (name, who it's for, benefit type, description):\n{options_desc}\n\n"
        f"Ask ONE short, simple clarifying question {lang_instruction} that would narrow this down "
        "(e.g. what specific kind of help they need). Read the descriptions above before deciding "
        "what to ask - if they span different PURPOSES, ask about purpose first, in plain everyday "
        "words a citizen would use ('aapko kis tarah ki madad chahiye').\n"
        "A PURPOSE is what the help is FOR - the actual thing that changes in the citizen's life "
        "(irrigating the field, buying equipment, building storage, planting an orchard, monthly "
        "income in old age, covering medical bills, paying school fees). It is NOT how the money "
        "is delivered. 'Subsidy', 'loan', 'grant', 'anudan' and 'sahayata' are FUNDING MECHANISMS, "
        "not purposes - almost every scheme here is one of them, so offering them as choices "
        "splits nothing. Found live (2026-09-02): a farmer was offered 'paise ki subsidy, khet ke "
        "equipment ke liye sahayata, ya storage/bagichi jaisi construction wali madad' and "
        "answered 'subsidy' - but all three of those ARE subsidies, so the answer narrowed nothing "
        "and the next reply just picked one scheme anyway. If you offer example options, they MUST "
        "be mutually exclusive: a citizen picking any one of them must rule the others out.\n"
        "Only drop to a scheme-specific operational "
        "detail (like a land size or an irrigation requirement) if every option above already "
        "shares the same purpose and only that operational detail actually separates them. Prefer "
        "a broad, low-sensitivity axis (age, class/degree, income, residence, purpose) over a "
        "narrow or sensitive one (disability, caste, marital status, gender) - ask about the "
        "latter only if every option above is gated on exactly that. It must be about something "
        "NOT already answered in the context above. Do "
        "not list scheme names. One sentence only."
    )
    reply = _invoke_reply(get_reply_llm(), [("system", PERSONA), ("human", prompt)], state.get("reply_lang"), label="clarify")
    return {
        "last_reply": reply,
        "pending_need_text": state.get("need_text"),
        "pending_need_domain": "scheme",
        "clarify_attempts": state.get("clarify_attempts", 0) + 1,
    }


def clarify_identity_node(state: ConvState):
    shortlist = state.get("shortlist") or []
    primary_need = (state.get("needs") or [None])[0]
    facts = _subject_facts_for_need(primary_need, state.get("profile", {}))
    need_text = primary_need["need_text"] if primary_need else state.get("need_text", "")
    top, gaps = _first_identity_gap(shortlist, facts, need_text)
    if top is None:  # defensive - routing already confirmed a gap exists somewhere in the top N
        top = shortlist[0]
    # Capped at 2, same limit as generate_node's QUESTION LIMIT - a single
    # scheme restricted on 3+ unconfirmed fields at once (e.g. gender AND
    # caste AND disability) would otherwise get jammed into one compound
    # gap_desc and asked as a single overloaded question. Any gap beyond the
    # first 2 simply isn't asked about THIS turn - once these are answered
    # and the shortlist is re-evaluated with the new facts, a remaining gap
    # (on this scheme or another) surfaces on a later turn same as normal.
    gaps = gaps[:2]
    gap_desc = ", ".join(f"{field.replace('_', ' ')}: {value.replace('|', ' or ')}" for field, value in gaps)
    lang_instruction = _lang_instruction(state.get("reply_lang"))

    # Fetch the scheme's OWN eligibility text, not just the coarse
    # structured tag - found live (2026-08-11): "Assistance for Daughter's
    # Marriage" requires the APPLICANT (a widowed parent) to be widowed, not
    # the daughter, but this node had no way to know that from the bare tag
    # alone and defaulted to assuming every gap on a family-member need
    # belongs to that family member. Verified directly: given the real
    # eligibility text, the model correctly identifies who a restriction
    # actually applies to - it's an information gap, not a reasoning one.
    lang = _source_lang(state.get("reply_lang"))
    svc = get_rag_service()
    elig_chunks = svc.scheme_detail(top["scheme_id"], lang, field_types=["eligibility"])
    if not elig_chunks and lang != "en":
        elig_chunks = svc.scheme_detail(top["scheme_id"], "en", field_types=["eligibility"])
    elig_text = elig_chunks[0][0] if elig_chunks else ""

    # The "could this actually be about the applicant, not the family member"
    # framing is ONLY relevant when the need genuinely IS about a family
    # member - found live (2026-08-11): "mai kheti karta hu... mere liye"
    # (unambiguously self, no third party mentioned at all) still got asked
    # "are you a farmer, or asking on behalf of someone else?" because this
    # clause used to be included unconditionally, mentioning the family-
    # member possibility even when about='self' and there was never any
    # ambiguity to begin with.
    if primary_need and primary_need.get("about") == "family_member":
        subject_desc = primary_need.get("subject_relation") or "the family member you mentioned"
        # Give the model what's already confirmed about the CITIZEN too, not
        # just the family member's facts - found live (2026-08-11): a citizen
        # already known (this same turn, from "...karta hu") to be male still
        # got asked "are you a widow of an ex-serviceman?" for a scheme
        # requiring a widowed applicant, because this node only ever looked
        # at the family member's facts and had no way to know the citizen's
        # own confirmed gender already ruled that out. It's a context gap,
        # not a reasoning one - the model already reasons correctly about
        # WHO an eligibility restriction targets (see the paragraph below);
        # it just never got told what's already known about the citizen.
        profile = state.get("profile", {})
        known_bits = [
            f"{f.replace('_', ' ')}: {profile[f]}"
            for f in ("gender", "caste", "marital_status", "age", "family_income", "residence", "education_level", "disability")
            if profile.get(f)
        ]
        known_desc = ", ".join(known_bits) if known_bits else "nothing yet"
        who_clause = (
            f"This need is for the citizen's {subject_desc}, not the citizen themselves. What's "
            f"already confirmed about the CITIZEN THEMSELVES (not {subject_desc}) so far: {known_desc}.\n"
            f"The eligibility text is the ONLY thing that decides who this specific restriction ({gap_desc}) "
            f"is actually about - do not guess or default, read it literally: if it describes a "
            f"requirement of the APPLICANT/PARENT (e.g. 'the applicant should be...'), that requirement "
            f"is about the CITIZEN, even though the overall need is for {subject_desc} - e.g. a scheme "
            "assisting a widowed parent with their child's marriage requires the PARENT to be widowed, "
            f"so that specific question must be addressed to the citizen, not the child. Only if the "
            f"eligibility text itself describes a requirement of the beneficiary/child/dependent does "
            f"this restriction mean {subject_desc}.\n"
            f"If you conclude the restriction is about the citizen, but what's already confirmed about "
            f"the citizen above directly contradicts it (e.g. it requires a woman/widow but the citizen "
            f"is already known to be male), then this scheme does not actually apply to this household - "
            f"say so briefly instead of asking a question whose answer is already known."
        )
    else:
        who_clause = (
            "Read the eligibility text to determine what it actually requires of the citizen "
            "(there is no family member involved in this need)."
        )
    prompt = (
        # Same context gap as clarify_node (2026-08-18): without the recent
        # conversation and durable notes, this node could ask about a
        # restriction the citizen had already addressed in a form the coded
        # facts didn't capture.
        f"{_format_conversation_history(state.get('conversation_history'))}"
        f"{_format_notes(state.get('conversation_notes'))}"
        f'The citizen asked about: "{state["need_text"]}"\n'
        f'The best-matching scheme is specifically restricted on: {gap_desc}. Its actual eligibility '
        f'text: "{elig_text}"\n\n'
        f"{who_clause}\n\n"
        f"Normally: ask ONE short, natural, non-intrusive question {lang_instruction} SPECIFICALLY "
        f"about the {gap_desc} criterion identified above - not a different, narrower requirement "
        "mentioned elsewhere in the eligibility text (e.g. a specific field of study/institution/"
        "sub-condition), even if that seems more concretely checkable. The eligibility text is only "
        "there to help phrase the question accurately and to determine who it's about (per the "
        "paragraph above) - it does not change WHICH criterion the question is about. Found live "
        "(2026-08-11): a generic 'give me a scholarship' request with an unconfirmed education_level "
        "gap got asked 'is your PhD in sociology/law/economics/...' instead - a wildly narrower, "
        "far less likely assumption than the actual gap (are they even in higher education at all), "
        "purely because that scheme's eligibility text happened to be dominated by that specific PhD "
        "requirement. "
        "If the identified gap is age or family income, ask for the actual VALUE openly ('aapki umar "
        "kitni hai?', 'aapke parivar ki saalana aay kitni hai?') - never a yes/no naming this "
        "scheme's own threshold ('kya aap 60 saal se upar hain?', 'kya aapki aay 2 lakh se kam "
        "hai?'), which both leads the answer and only resolves THIS one scheme, where the real "
        "number resolves every other age- or income-gated scheme in the shortlist at once. "
        "If the identified gap is caste or marital status - fields with SEVERAL possible values "
        "(general/OBC/SC/ST/PVTG for caste; married/never-married/widowed/divorced/separated for "
        "marital status) - phrase the question OPEN-ENDED, asking what category/status the RELEVANT "
        "PERSON belongs to, NOT a presumptuous yes/no naming the one specific value THIS scheme "
        "happens to require (avoid 'are you SC?'/'are you widowed?' framings - that presumes an "
        "answer and reads as leading). This open-ended phrasing follows the SAME who-it's-addressed-"
        "to rule as everything else here - if the restriction is about the citizen, ask 'aap kis "
        "caste category se hain?'; if it's about a family member (e.g. their son), ask about THAT "
        "person by name/relation, e.g. 'aapke bete ki caste category kya hai?', never a bare 'aap/"
        "aapki' that would wrongly imply it's asking about the citizen themselves. Learning their "
        "actual category also resolves eligibility for every OTHER caste- or marital-status-"
        "restricted scheme in the shortlist from this one answer, not just this scheme. For every "
        "other gap field (residence, education level, disability, gender), a direct yes/no about "
        "the specific requirement is fine, since those are simple binary confirmations, not "
        "multi-way categories. "
        "Address the question to whichever person you determined above the restriction "
        "is actually about, and mention that general options exist either way. Do NOT name the "
        "specific scheme. "
        "But if the paragraph above says this scheme doesn't actually apply, say that briefly instead "
        f"{lang_instruction} and mention general options exist - do not ask a question in that case. "
        "Likewise, if the conversation or noted facts above already answer the question you would "
        "ask, do not ask it - use that answer directly instead. "
        "One or two sentences either way."
    )
    reply = _invoke_reply(get_reply_llm(), [("system", PERSONA), ("human", prompt)], state.get("reply_lang"), label="clarify_identity")
    return {
        "last_reply": reply,
        "pending_need_text": state.get("need_text"),
        "pending_need_domain": "scheme",
        # Pin the scheme this question was asked ABOUT (see ConvState.
        # pending_clarify_scheme_ids) - `top` is the specific shortlist entry
        # whose unconfirmed gap produced gap_desc, so it is exactly the scheme
        # the citizen's answer is meant to resolve, and exactly the one that
        # must not be allowed to vanish from the next turn's shortlist.
        "pending_clarify_scheme_ids": [top["scheme_id"]],
        "clarify_attempts": state.get("clarify_attempts", 0) + 1,
    }


def stage_b_node(state: ConvState):
    shortlists = state.get("shortlists") or [{"need": None, "shortlist": state.get("shortlist") or []}]
    lang = _source_lang(state.get("reply_lang"))
    field_types = FIELD_TYPES_BY_INTENT.get(state.get("intent"), ["eligibility", "benefits"])

    svc = get_rag_service()
    chunks = []
    already_shown = list(state.get("already_shown_scheme_ids") or [])
    shown_schemes = []
    # Fetch detail for every need's top schemes, not just the first - a
    # message stating two unrelated needs (2026-08-10) used to only ever see
    # the first need's schemes in the final answer, since this only looped
    # over the single `shortlist`. shown_schemes records one entry per need
    # so a later followup ("pehli scheme"/"dusri scheme") has something real
    # to resolve against, instead of a single stale shown_scheme_id that only
    # ever reflected the first need.
    profile = state.get("profile", {})
    for entry in shortlists:
        need = entry.get("need") or {}
        facts = _subject_facts_for_need(need, profile)
        # _prioritize_SENSITIVE_gap_free, not the plain version (2026-09-02) -
        # this is the SAME bug _prioritize_sensitive_gap_free's own docstring
        # already documents finding and fixing on 2026-08-24, just never
        # propagated to this call site. Found live again here: a scholarship
        # search where every candidate has SOME gap (education_level is
        # almost always unconfirmed for a first message) left the plain
        # gap-free/gapped split a no-op - its "gap-free" bucket was empty, so
        # rank order passed through untouched, and a disability-gated
        # scholarship sitting at/near the top got fetched as one of
        # generate_node's top-3 schemes anyway. generate_node then had no
        # choice but to reason about IT specifically, leading with "is your
        # son disabled" ahead of a broader, non-sensitive gap (which course/
        # college) sitting on a DIFFERENT candidate in the same shortlist.
        # _prioritize_sensitive_gap_free only demotes a scheme gapped on a
        # SENSITIVE field (gender/caste/marital_status/disability-adjacent
        # occupation groups) - one gapped only on a broad fact like
        # education_level stays in its normal rank position, so it still
        # correctly separates "everything has some gap" from "this one has a
        # SENSITIVE gap", which is the actual distinction that matters here.
        top = _prioritize_sensitive_gap_free(entry["shortlist"] or [], facts, need.get("need_text", ""), 3)
        # Record EVERY scheme actually presented for this need, not just the
        # top one - found live (2026-08-11): a single need surfacing 3
        # candidate schemes together (the normal case, not just the multi-
        # need case) left shown_schemes with only 1 entry even though all 3
        # were fetched and shown in the answer below, so "dusri scheme"
        # (second scheme) had nothing real to resolve against.
        for s in top:
            shown_schemes.append({
                "scheme_id": s["scheme_id"],
                "scheme_name": s["scheme_name"],
                "need_text": need.get("need_text", ""),
            })
        for s in top:
            detail = svc.scheme_detail(s["scheme_id"], lang, field_types=field_types)
            if not detail and lang != "en":
                detail = svc.scheme_detail(s["scheme_id"], "en", field_types=field_types)
            chunks.extend(detail)
        if top and top[0]["scheme_id"] not in already_shown:
            already_shown.append(top[0]["scheme_id"])

    # Narrow chunk_cache carryover to just the named scheme(s) when this
    # need was a genuine, unambiguous name lookup (2026-08-25, found live) -
    # previously ALWAYS None here, on the reasoning that a fresh need is "a
    # genuinely fresh, possibly multi-scheme answer - no narrowing to carry
    # over". That's right for a generic need, but wrong for "am I eligible
    # for Indira Mahila Shakti Udyam Protsahan Yojana" asked for the FIRST
    # time (never previously shown, so stage_b_followup_node's own
    # focused_scheme_ids never applies either) - without narrowing here,
    # whatever OTHER schemes were discussed several turns earlier stayed in
    # generate_node's prompt via chunk_cache, unfiltered, right alongside
    # the correctly-retrieved named scheme - verified live: the model
    # answered about the named scheme using a DIFFERENT, stale scheme's
    # eligibility text still sitting in that same prompt.
    named_scheme_ids = [sid for entry in shortlists for sid in (entry.get("named_scheme_ids") or [])]

    return {
        "detail_chunks": chunks,
        "chunk_cache": _update_chunk_cache(state.get("chunk_cache"), chunks),
        "shown_schemes": shown_schemes,
        "shown_scheme_id": shown_schemes[0]["scheme_id"] if shown_schemes else state.get("shown_scheme_id"),
        "shown_scheme_name": shown_schemes[0]["scheme_name"] if shown_schemes else state.get("shown_scheme_name"),
        "shown_scheme_lang": lang,
        "already_shown_scheme_ids": already_shown,
        "clarify_attempts": 0,  # a real answer is being delivered - the unresolved streak ends here
        "focused_scheme_ids": named_scheme_ids or None,
    }


def show_more_node(state: ConvState):
    """'What other schemes am I eligible for' - a distinct request from a
    single-scheme followup (turn_type='followup_same_scheme'), answered from
    the shortlist already computed by the last Stage A run rather than a
    fresh search, since that data already exists (see ConvState.shortlist's
    docstring). Excludes every scheme already shown this conversation, not
    just the current one, so asking "what else" repeatedly doesn't repeat
    an earlier answer."""
    already_shown = set(state.get("already_shown_scheme_ids") or [])
    remaining = [s for s in (state.get("shortlist") or []) if s["scheme_id"] not in already_shown][:3]
    lang = _source_lang(state.get("reply_lang"))
    field_types = FIELD_TYPES_BY_INTENT.get(state.get("intent"), ["eligibility", "benefits"])

    svc = get_rag_service()
    chunks = []
    for s in remaining:
        detail = svc.scheme_detail(s["scheme_id"], lang, field_types=field_types)
        if not detail and lang != "en":
            detail = svc.scheme_detail(s["scheme_id"], "en", field_types=field_types)
        chunks.extend(detail)

    new_already_shown = list(already_shown)
    if remaining and remaining[0]["scheme_id"] not in new_already_shown:
        new_already_shown.append(remaining[0]["scheme_id"])

    shown_schemes = [
        {"scheme_id": s["scheme_id"], "scheme_name": s["scheme_name"], "need_text": state.get("need_text", "")}
        for s in remaining
    ]

    return {
        "detail_chunks": chunks,
        "chunk_cache": _update_chunk_cache(state.get("chunk_cache"), chunks),
        "shown_schemes": shown_schemes or state.get("shown_schemes"),
        "shown_scheme_id": remaining[0]["scheme_id"] if remaining else state.get("shown_scheme_id"),
        "shown_scheme_name": remaining[0]["scheme_name"] if remaining else state.get("shown_scheme_name"),
        "shown_scheme_lang": lang,
        "already_shown_scheme_ids": new_already_shown,
        "focused_scheme_ids": None,  # showing a fresh batch of alternatives - no narrowing to carry over
    }


def stage_b_followup_node(state: ConvState):
    lang = state.get("shown_scheme_lang", "en")
    field_types = FIELD_TYPES_BY_INTENT.get(state.get("intent"), ["eligibility", "benefits"])

    # Resolve which shown scheme(s) this followup actually means, rather than
    # always defaulting to a single shown_scheme_id - found live (2026-08-10):
    # when several schemes were shown at once, "pehli scheme"/"education wali"
    # style followups had nothing to resolve against and silently answered
    # about the wrong (stale, unrelated) scheme. followup_target is resolved
    # by extract_need_node from the actual message text against the real
    # list of shown schemes, not guessed positionally here.
    shown_schemes = state.get("shown_schemes") or []
    target_name = state.get("followup_target")
    targets = [s for s in shown_schemes if s["scheme_name"] == target_name] if target_name else []
    if not targets:
        targets = shown_schemes or [{"scheme_id": state["shown_scheme_id"], "scheme_name": state.get("shown_scheme_name")}]

    svc = get_rag_service()

    # Re-verify eligibility against whatever's been learned SINCE this
    # scheme was first shown, before presenting it again - a followup never
    # re-runs Stage A, so `targets` can be stale. Found live (2026-08-10): a
    # citizen who explicitly said "no disability" kept being shown a
    # disability-restricted scheme turn after turn, because nothing ever
    # re-checked it against that new fact.
    primary_need = (state.get("needs") or [None])[0]
    profile = state.get("profile", {})
    facts = _subject_facts_for_need(primary_need, profile)
    citizen = CitizenContext(
        caste=facts.get("caste"), gender=facts.get("gender"), marital_status=facts.get("marital_status"),
        age=facts.get("age"), family_income=facts.get("family_income"), residence=facts.get("residence"),
        education_level=facts.get("education_level"), disability=facts.get("disability"),
    )
    # For a family-member need, also check the citizen's OWN profile as an
    # alternate identity - mirrors stage_a_node's alt_citizen (same fail-open
    # "exclude only if NEITHER identity fits" reasoning). Found live
    # (2026-08-11): this re-check only ever looked at the family member's
    # facts, so a scheme whose applicant-only requirement was already
    # contradicted by the citizen's own known profile (e.g. married, not
    # widowed) still came back "still eligible" on a followup, even though a
    # fresh clarify_identity_node evaluation would correctly reject it.
    alt_citizen = None
    if primary_need and primary_need.get("about") == "family_member":
        alt_citizen = CitizenContext(
            caste=profile.get("caste"), gender=profile.get("gender"), marital_status=profile.get("marital_status"),
            age=profile.get("age"), family_income=profile.get("family_income"), residence=profile.get("residence"),
            education_level=profile.get("education_level"), disability=profile.get("disability"),
        )
    still_eligible, now_excluded = [], []
    for t in targets:
        meta = svc._meta_by_scheme_id.get(t["scheme_id"])
        fits = meta is None or eligibility_verdict(meta, citizen)[0] or (
            alt_citizen is not None and eligibility_verdict(meta, alt_citizen)[0]
        )
        if fits:
            still_eligible.append(t)
        else:
            now_excluded.append(t["scheme_id"])

    # If the citizen explicitly asked about ONE specific scheme (via
    # followup_target) and it turns out excluded, still fetch and present
    # ITS OWN detail so the answer can explain why - found live (2026-08-11):
    # "3rd scheme ki eligibility batao" asked about a scheme that's excluded
    # once the citizen's gender contradicts it, still_eligible came back
    # empty, and the answer silently talked about two OTHER schemes instead
    # of ever addressing the one actually asked about. Only falls back to
    # "explain the excluded one" when a SPECIFIC target was named - a plain
    # followup with no named target should still just quietly drop anything
    # excluded, not resurface it.
    to_fetch = still_eligible if (still_eligible or not target_name) else targets

    chunks = []
    for t in to_fetch:
        detail = svc.scheme_detail(t["scheme_id"], lang, field_types=field_types)
        if not detail and lang != "en":
            detail = svc.scheme_detail(t["scheme_id"], "en", field_types=field_types)
        chunks.extend(detail)

    # Always carry a couple of ALTERNATIVES alongside the focused scheme,
    # unless the citizen explicitly named one specific scheme to talk about
    # (2026-09-02). Until now this node fetched the shown scheme's text and
    # NOTHING else, which is fine while that scheme still fits - but leaves
    # generate_node with literally no other scheme in context the moment it
    # doesn't, so the only reply it can physically produce is "yeh fit nahi
    # baithega" plus a promise to look at other options it has no means to
    # keep.
    #
    # Found live twice on consecutive days, same shape, different schemes:
    # "bete ne 12th pass ki hai, scholarship chahiye" locked onto a PhD-
    # abroad scholarship (age 21-34) in one session and a class-10/12 talent-
    # exam scholarship in another, then spent THREE turns each re-explaining
    # that same non-fit - including in reply to a bare "btao" ("just tell
    # me"), which is about as direct a request for the alternatives as a
    # citizen can make. Neither scheme could be dropped mechanically either:
    # eligibility_verdict is fail-open, and the facts that actually rule
    # these out (the son's age; "currently studying in class 10/12") are
    # respectively unconfirmed and not represented in the structured schema
    # at all - so the re-verification above correctly leaves them in place.
    # Giving the node somewhere to GO is the fix that doesn't depend on
    # detecting the non-fit mechanically.
    #
    # _prioritize_sensitive_gap_free, matching stage_b_node, so a narrow
    # sensitive-gated scheme isn't offered as the fallback - but otherwise
    # retrieval rank is preserved, since the fallback still has to be a
    # RELEVANT scheme (see that helper's note on the reverted three-tier
    # variant, which promoted unrestricted-but-irrelevant schemes). Skipped
    # entirely when target_name is set, so an
    # explicit "tell me about the last scheme" still narrows exactly as
    # before (see focused_scheme_ids below).
    if not target_name:
        target_ids = {t["scheme_id"] for t in to_fetch}
        need_text = primary_need.get("need_text", "") if primary_need else state.get("need_text", "")
        ranked = _prioritize_sensitive_gap_free(state.get("shortlist") or [], facts, need_text, 10)
        alternatives = [s for s in ranked if s["scheme_id"] not in target_ids][:2]
        for a in alternatives:
            detail = svc.scheme_detail(a["scheme_id"], lang, field_types=field_types)
            if not detail and lang != "en":
                detail = svc.scheme_detail(a["scheme_id"], "en", field_types=field_types)
            chunks.extend(detail)

    # Purge any now-disconfirmed scheme from the PERSISTENT rolling cache -
    # it can still be fetched fresh above to explain THIS turn's answer, but
    # shouldn't linger in the cache afterward as if it were still a live
    # candidate.
    cache = [(doc, meta) for doc, meta in (state.get("chunk_cache") or []) if meta.get("scheme_id") not in now_excluded]
    cacheable_ids = {t["scheme_id"] for t in still_eligible}
    cacheable_chunks = [c for c in chunks if c[1].get("scheme_id") in cacheable_ids]

    # When the citizen explicitly narrowed to ONE specific scheme
    # (followup_target matched something), tell generate_node to restrict
    # itself to just that scheme - found live (2026-08-11): "tell me about
    # the last scheme" correctly narrowed THIS turn's own fetch, but
    # chunk_cache still held the other 2 previously-shown schemes from the
    # prior turn, and generate_node's "fill in recent context" merge
    # silently reintroduced them, defeating the narrowing entirely. Always
    # explicitly set (never left to the shallow-merge default) so a
    # narrowing from an earlier turn can't linger onto a later turn that
    # didn't narrow anything.
    focused_scheme_ids = [t["scheme_id"] for t in to_fetch] if target_name else None
    return {
        "detail_chunks": chunks,
        "chunk_cache": _update_chunk_cache(cache, cacheable_chunks),
        "focused_scheme_ids": focused_scheme_ids,
    }


def _structured_facts_summary(meta):
    """The free-text field chunks (description/benefits/eligibility prose)
    don't always mention every structured eligibility fact myScheme's API
    provides - e.g. a scheme's caste facet can include OBC/SC/ST while its
    free-text eligibility description only talks about income and
    institution type, never mentioning caste at all (verified 2026-08-05
    on Mukhya Mantri Sarvajan Uchcha Shiksha Chhatravriti Yojana). Stage A's
    filter already used these structured facts to decide this scheme is
    relevant - generation needs to see them too, or it answers blind to
    exactly the signal that made the scheme worth showing.

    Identity dimensions (caste/gender/marital_status) are ALWAYS stated
    explicitly, including when unrestricted ("open to all") - found live
    (2026-08-05): omitting caste entirely when it's 'all' left the LLM
    seeing silence on both the structured data AND the free text (which
    also doesn't mention caste), and it concluded "doesn't mention OBC"
    (hedging) instead of "no caste restriction, OBC qualifies" - silence
    is not the same as "no restriction" unless stated as such.
    """
    parts = []
    for key in ("caste", "gender", "marital_status"):
        val = meta.get(key) or "all"
        parts.append(
            f"{key.replace('_', ' ')}: open to all"
            if val == "all"
            else f"{key.replace('_', ' ')}: {val.replace('|', ', ')}"
        )
    for key in ("occupation_target_group", "benefit_type_fine"):
        val = meta.get(key)
        if val:
            parts.append(f"{key.replace('_', ' ')}: {val.replace('|', ', ')}")
    if meta.get("family_income_ceiling"):
        parts.append(f"family income ceiling: Rs {meta['family_income_ceiling']}")
    if meta.get("age_min") is not None or meta.get("age_max") is not None:
        parts.append(f"age range: {meta.get('age_min')}-{meta.get('age_max')}")
    return "; ".join(parts)


def _format_conversation_history(history):
    if not history:
        return ""
    lines = [f'Citizen: "{turn["user"]}"\nYou: "{turn["bot"]}"' for turn in history]
    return "Recent conversation (for context only - answer the CURRENT question, don't repeat old answers):\n" + "\n\n".join(lines) + "\n\n"


def _format_profile(profile):
    if not profile:
        return ""
    facts = "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in profile.items())
    # "never ask again" instruction added 2026-08-25 (found live) - a citizen
    # stated "meri umar lagbhag 36 saal hai", the reply correctly used
    # age=36 in its own eligibility reasoning ("aap 36 hain, 40+ chahiye"),
    # then STILL asked "kya aapki umar 40 saal poori ho gayi hai, ya abhi 36
    # hi hai?" - restating the exact fact it had JUST used, as if
    # unconfirmed. _format_notes already carries this instruction for
    # conversation_notes; this block never had the equivalent one, even
    # though it's fed into the exact same prompts.
    return (
        f"What we know about this citizen: {facts} (already stated - never ask the citizen to "
        "confirm/restate any of this)\n\n"
    )


def _format_notes(notes):
    """conversation_notes as a prompt block - the durable free-form memory
    that outlives the rolling history window (fed by both ProfileExtraction.
    new_notes and NeedExtraction.stated_facts), so anything recorded here is
    already-given information a reply must never ask for again."""
    if not notes:
        return ""
    lines = "\n".join(f"- {n}" for n in notes)
    return (
        "Other facts noted earlier in this conversation (treat these as already answered - "
        f"never ask about any of them again):\n{lines}\n\n"
    )


# Appended to BOTH of generate_node's rewrite prompts (2026-09-01). Those are
# the longest prompts in this file - the entire original prompt plus a
# correction - and a rewrite instruction naturally reads as being about the
# CONTENT of the reply, which makes it easy to treat the reply itself as
# something already established and only re-emit the flags. That is exactly
# what was seen live: a generate_fix response carrying both booleans and no
# `reply` at all. Stated explicitly here rather than relying on the field
# description alone, since by this point the field descriptions are far up a
# very long prompt.
_REWRITE_FIELD_REMINDER = (
    "Return the COMPLETE rewritten reply text in the `reply` field - the rewrite is the reply "
    "itself, not just a change to the flags. A response without `reply` is unusable: it is the "
    "only part the citizen ever sees."
)


class SchemeAnswer(_UnwrapsToolEnvelope):
    """Structured wrapper around generate_node's reply purely to know
    whether it asked a clarifying question - added 2026-08-16 so
    pending_need_text can be set the same way clarify_node/
    clarify_identity_node already do, giving a terse follow-up reply next
    turn something structured to anchor to (see generate_node's return)."""

    # `reply` is the ONLY strictly-required field here; both booleans below
    # carry defaults (2026-09-01). Found live in the server log: a
    # generate_fix call came back as
    # {'asked_clarifying_question': ..., 'named_scheme_has_unconfirmed_condition': False}
    # with `reply` absent entirely - the model answered the two cheap metadata
    # questions and dropped the expensive string. That fired on generate_fix
    # specifically, which is the longest prompt in this file (the full
    # generate prompt with a correction appended), and right after
    # named_scheme_has_unconfirmed_condition took this model from two fields
    # to three - more required fields on the longest prompt is exactly the
    # pressure that makes a model skip the payload.
    #
    # Defaults don't prevent that case (a missing `reply` still has to fail -
    # there's nothing safe to substitute for the citizen's actual answer, so
    # _invoke_structured's retry remains correct there). What they DO prevent
    # is the reverse failure: a payload with a perfectly good `reply` being
    # thrown away, and a second full-size call paid for, purely because a
    # boolean was omitted. The defaults are the conservative reading in both
    # cases - asked_clarifying_question=False means "treat this as a complete
    # answer" (so pending_need_text isn't left dangling on a reply that never
    # asked anything), and named_scheme_has_unconfirmed_condition=False means
    # "don't trigger the hedge-rewrite retry", i.e. an omitted flag never
    # invents extra work, it just declines to claim a violation it can't see.
    reply: str = Field(..., description="The actual reply to send to the citizen - identical in content to what a plain text reply would be. ALWAYS include this field; it is the only thing the citizen actually receives.")
    asked_clarifying_question: bool = Field(
        False,
        description=(
            "True if `reply` asks the citizen to confirm/provide ANY information needed to "
            "determine eligibility (per QUESTION LIMIT above) - i.e. it is NOT yet a complete, "
            "final answer. False if `reply` is a complete answer with no outstanding question "
            "(eligibility confirmed, or plainly stating nothing matches)."
        ),
    )
    named_scheme_has_unconfirmed_condition: bool = Field(
        False,
        description=(
            "True ONLY if `reply` names/recommends a scheme (asked_clarifying_question=False) whose "
            "real eligibility text (not just the coarse structured tag) contains a condition - "
            "numeric/quantitative (land size in hectares, income amount, age cutoff, family size, "
            "distance) or otherwise - that is NOT already confirmed in 'What we know about this "
            "citizen'/the conversation, REGARDLESS of whether `reply` phrases it as a caveat/hedge "
            "('agar aapki zameen 1.5 hectare se kam hai to yeh fit nahi karegi') rather than a "
            "direct question. False if every condition the named scheme's eligibility text states "
            "is already confirmed, or if `reply` didn't name any scheme."
        ),
    )


def generate_node(state: ConvState):
    # this turn's own retrieval is authoritative and always included; the
    # rolling cache only fills in recently-discussed schemes/fields THIS
    # turn's retrieval didn't happen to fetch (e.g. this turn only pulled
    # "documents" but the citizen's question also implicitly touches
    # "eligibility" discussed two turns ago) - current-turn chunks win on
    # any (scheme_id, field_type, lang) key collision.
    current = state.get("detail_chunks") or []
    current_keys = {_chunk_key(meta) for _, meta in current}
    cached_extra = [(doc, meta) for doc, meta in (state.get("chunk_cache") or []) if _chunk_key(meta) not in current_keys]

    # When the citizen just explicitly narrowed to ONE specific scheme (see
    # stage_b_followup_node's focused_scheme_ids), restrict the cache
    # fill-in to that same scheme - found live (2026-08-11): "tell me about
    # the last scheme" correctly narrowed this turn's own fetch, but the
    # cache still held the other previously-shown schemes and got merged
    # back in here, silently undoing the narrowing the citizen just asked
    # for. Every other path leaves this unset, so the normal "fill in
    # recent context across multiple schemes" behavior is unaffected.
    focused_scheme_ids = state.get("focused_scheme_ids")
    if focused_scheme_ids:
        cached_extra = [(doc, meta) for doc, meta in cached_extra if meta.get("scheme_id") in focused_scheme_ids]

    chunks = current + cached_extra

    structured_by_scheme = {}
    for _, meta in chunks:
        structured_by_scheme.setdefault(meta["scheme_name"], meta)
    structured_text = "\n".join(
        f"[{name}] structured eligibility data: {_structured_facts_summary(meta)}"
        for name, meta in structured_by_scheme.items()
    )

    chunks_text = "\n\n".join(
        f"[{meta['scheme_name']} / {meta['field_type']}]\n{doc}" for doc, meta in chunks
    )

    # Pre-synthesized short cards (name + who it's for + one-line
    # description), reused from Stage A's own search index rather than
    # having the model improvise a summary from the full eligibility/
    # benefit text below - see "RESPONSE ORDER"/clarifying-question
    # instructions further down for when this gets used INSTEAD of the
    # full text, not alongside it.
    svc = get_rag_service()
    discovery_cards_text = "\n".join(
        f"[{name}] {svc.get_discovery_card(meta['scheme_id'], meta['lang'])}"
        for name, meta in structured_by_scheme.items()
    )

    lang_instruction = _lang_instruction(state.get("reply_lang"))

    needs = state.get("needs") or [{"need_text": state.get("need_text"), "about": "self", "subject_relation": None}]
    if len(needs) > 1:
        needs_lines = []
        for i, n in enumerate(needs, 1):
            who = f" (for the citizen's {n.get('subject_relation') or 'family member'}, not the citizen themselves)" if n.get("about") == "family_member" else ""
            needs_lines.append(f'{i}. "{n["need_text"]}"{who}')
        question_block = (
            "Citizen's CURRENT questions (there is more than one - answer EACH separately and "
            "clearly under its own heading, do not merge them into one answer):\n" + "\n".join(needs_lines)
        )
    else:
        who = f" (for the citizen's {needs[0].get('subject_relation') or 'family member'}, not the citizen themselves)" if needs[0].get("about") == "family_member" else ""
        question_block = f'Citizen\'s CURRENT question: "{needs[0]["need_text"]}"{who}'

    # need_text is a deliberately distilled search phrase (see NeedItem's own
    # field description) - it can lose specific facts the citizen just
    # confirmed, especially ones about a referenced third party supporting
    # THEIR OWN need (e.g. "but my parents are construction worker" for a
    # scheme requiring the applicant's PARENT to be one) - there's no
    # structured field for that (it isn't the need's own subject, so
    # subject_facts doesn't apply, and it isn't the citizen's own fact
    # either). Found live (2026-08-11): need_text collapsed that exact
    # message to 'higher studies scholarship', and the answer confidently
    # denied eligibility, having never seen the confirmation at all. Passing
    # the literal raw message lets the model use whatever specific facts
    # were actually stated, however they relate to the scheme's real
    # eligibility text, without needing a new field for every possible kind
    # of referenced third party.
    question_block += f'\nCitizen\'s exact words this turn: "{state.get("raw_message", "")}"'

    # Consolidated 2026-08-17 from six separately-labeled rules (RESPONSE
    # ORDER/QUESTION LIMIT/QUESTION PRIORITY/SCHEME DETAIL LEVEL/ANSWER
    # EARLY/HARD STOP) into two sections below - the accumulated rule count
    # itself was making replies read like a checklist being executed rather
    # than a person reasoning, which is a real, separate complaint from any
    # single rule being wrong. Nothing here removes a protection; the
    # concrete bugs each rule was fixing are preserved below in comments
    # (not sent to the model - it doesn't need bug history, just the
    # current rule) so the reasoning isn't lost, only the prompt bulk:
    #   - RESPONSE ORDER (2026-08-17): question buried after a long list of
    #     scheme names/conditions read as an afterthought.
    #   - QUESTION LIMIT (2026-08-14): a vague scholarship need surfaced
    #     FIVE questions (residence/age/class/disability/income) at once.
    #   - QUESTION PRIORITY (2026-08-16): "paisa chahiye beti ki padhai ke
    #     liye" opened with "does she have a 40%+ disability certificate"
    #     purely because one of three matched schemes was disability-gated.
    #   - SCHEME DETAIL LEVEL: full eligibility/benefit text for schemes
    #     whose fit isn't even confirmed yet buried the actual question.
    #   - ANSWER EARLY / HARD STOP (2026-08-17): a business-loan need was
    #     asked about across SEVEN consecutive turns without ever landing
    #     on a recommendation - one question even repeated something
    #     already answered several turns earlier, once it scrolled out of
    #     conversation_history's rolling window.
    generate_clarify_rounds = state.get("generate_clarify_rounds", 0)
    hard_stop_block = (
        "HARD STOP: you've already asked this many rounds of questions about this need without "
        "landing on an answer. Give your FINAL best-effort answer now - recommend the most "
        "likely-fitting scheme(s) from what's already known, caveat anything unconfirmed, and do "
        "not ask again, even a reasonable one.\n\n"
        if generate_clarify_rounds >= 2 else ""
    )

    prompt = (
        "Answer the citizen's question using ONLY the information below (both the "
        "structured eligibility data AND the scheme text) - never invent eligibility criteria, "
        "amounts, or steps not present in either.\n\n"
        "GROUNDING: the structured eligibility data is a COARSE tag (e.g. one caste value) - the "
        "scheme text is the actual, complete, government-written eligibility description, and is "
        "authoritative when the two disagree (say so plainly rather than calling it a confident "
        "match off the tag alone). A chunk labeled 'exclusions' is JUST AS eligibility-relevant as "
        "one labeled 'eligibility' - it's where termination/disqualification conditions live (e.g. "
        "'if landholding exceeds the small/marginal category, pension stops') - if it describes a "
        "condition genuinely unconfirmed about this citizen that would end their eligibility, treat "
        "it exactly like an eligibility gap: ask about it if it's the only thing stopping a "
        "confident answer, don't silently skip it just because it wasn't under the 'eligibility' "
        "label. Never assume or state the citizen's OWN caste, disability, "
        "gender, or any other identity fact unless it's explicitly listed in 'What we know about "
        "this citizen' below - if it isn't confirmed, say plainly that it isn't, rather than "
        "guessing or labeling them into a category that was never stated. If the citizen has "
        "already stated a fact - in the profile, the noted facts, or the conversation above - "
        "that clearly excludes a scheme, DO NOT include that scheme in your list of options at "
        "all - check this against the FULL scheme text, not just the coarse structured tag, since "
        "the tag can be wrong or too broad (e.g. tagged for a whole caste category when the real "
        "text restricts it to specific sub-groups the citizen isn't part of). A brief one-line "
        "mention that it doesn't apply is fine if directly relevant, but never present an "
        "already-excluded scheme as one of the options, even with a caveat attached - and don't "
        "keep re-asking something they've effectively already answered either. If a bare "
        "'yes/no' this turn is ambiguous (your last reply asked about two or more separate things "
        "at once), ask which one it's answering rather than assuming. If NONE of the schemes match "
        "what's known, say so plainly rather than repeating the same conditional list turn after "
        "turn.\n\n"
        "THE SAME FAIL-OPEN RULE APPLIES TO EXCLUDING A SCHEME, NOT JUST TO CONFIRMING ONE: never "
        "state that a scheme does NOT fit the citizen because of an identity fact or condition "
        "(gender, disability, being a beneficiary of another named scheme, marital status, "
        "anything about who they are) that the citizen never actually stated - missing information "
        "excludes nothing, in either direction. Found live (2026-09-03): a citizen who never "
        "mentioned their gender at all was told a scheme was 'un ladkiyon ke liye' (for those "
        "girls) and didn't fit them - that scheme's own structured tag happened to include 'woman' "
        "as ONE coarse label among others, but its real eligibility text never mentioned gender at "
        "all, and the citizen was never asked. The structured tag is a hint that a fact might be "
        "worth confirming, never proof of what the citizen's own gender, disability, marital "
        "status, or any other identity fact IS or ISN'T - the GROUNDING rule above (real text is "
        "authoritative over the coarse tag) applies here too. The same live case also rejected the "
        "scheme for not being 'a beneficiary under the Palanhar Scheme' - a real condition in the "
        "scheme's own text, but again never asked about. A condition that genuinely isn't stated "
        "either way is UNRESOLVED, not failed: either ask about it if it's plausible and would "
        "change the answer, or if you judge it too narrow/unlikely to be worth a question, say the "
        "fit is uncertain because of that specific unconfirmed condition - never declare a firm "
        "'this doesn't fit you' for a fact you were never told.\n\n"
        "NEVER PROMISE, ALWAYS DELIVER: if you conclude a scheme does NOT fit the citizen, you must "
        "move to a concrete alternative in the SAME reply - name an actual scheme from the text "
        "below that does fit, or state plainly that nothing here fits. Saying some version of "
        "'don't worry, other options exist, let me find the right one' and then stopping is a "
        "non-answer: the citizen cannot act on it, and you will be given the same scheme again next "
        "turn, so the promise never gets kept. Found live twice (2026-09-02): a 12th-pass student's "
        "scholarship request spent THREE consecutive turns explaining that one scheme didn't fit "
        "and pledging to look at undergraduate options, once even in reply to a bare 'btao' ('just "
        "tell me') - no alternative was ever named. If you genuinely need one more fact to pick "
        "between the alternatives you were given, ask for that ONE fact and nothing else - but do "
        "not re-explain the ruled-out scheme again to do it, and never re-litigate a scheme you "
        "have already told this citizen does not fit them; treat it as closed and do not bring it "
        "up a second time.\n\n"
        "HAVING THE CONVERSATION: have this like a real conversation, not an intake form - but "
        "(2026-08-19, reversing an earlier 'lead with scheme names' rule) do NOT name or list ANY "
        "scheme while you still need to ask something. The test is NOT how many candidates remain - "
        "it's whether THIS reply contains a question at all. If it does, zero scheme names appear "
        "anywhere in it, full stop, even if only 1 or 2 candidates are left. Found live (2026-08-19): "
        "'kheti karta hu, 60 saal ka hu' surfaced exactly two candidate pension schemes by name, "
        "amounts and all, then STILL asked about BPL status to know which one actually applies - "
        "naming schemes while a real eligibility question (one that could rule either of them out) "
        "is still open is exactly the mistake this rule exists to prevent, regardless of the "
        "candidate count.\n\n"
        "Once a reply asks NOTHING, name exactly the ONE scheme that most clearly and CONFIRMEDLY "
        "fits - not two. Found live again the same day: naming ONE clearly-fitting scheme correctly "
        "avoided a question this time, but then hedged in a SECOND scheme framed as 'you could also "
        "try this IF your income is under X' - income was never actually confirmed, so this is the "
        "exact same mistake wearing a conditional instead of a question mark: a scheme whose fit "
        "isn't confirmed still doesn't belong in this answer, whether it's attached to a question or "
        "to an 'if this applies to you' hedge. Only mention a second scheme if it is ALSO fully "
        "confirmed-fitting with nothing conditional about it (rare - most of the time there's one "
        "clear answer). This applies EQUALLY when there's only ONE scheme in play, not just when a "
        "second one is being added - found live (2026-09-01): a farmer who'd only confirmed 'khud ka "
        "khet hai' (has his own field, no size given) was told 'Kantedar Tarbandi Yojana fits you' "
        "with a hedge tacked on ('agar aapki zameen 1.5 hectare se kam hai to yeh fit nahi karegi - "
        "bata dena') instead of just ASKING the land size before naming anything - the scheme's real "
        "eligibility text has a hard 1.5-hectare minimum, so land size wasn't a minor refinement, it "
        "was the exact fact that determines fit, and it was still unconfirmed. A numeric/quantitative "
        "threshold in the scheme text (land size in hectares, income amount, age cutoff, family "
        "size, distance, etc.) is exactly as much an eligibility gap as caste/gender/marital status "
        "are - if the citizen's own value for that specific number isn't confirmed above, that is "
        "the question to ask, not a caveat to hand off for the citizen to self-check. A partial "
        "answer to an earlier multi-part question (e.g. 'haan khud ka khet hai' answering only the "
        "'do you own land' half of 'do you own land, and how many hectares') leaves the other half "
        "still open - treat it as still unconfirmed, don't treat the whole question as answered. "
        "If the citizen wants to see other options, they'll ask ('koi aur scheme?') "
        "and that's handled separately - don't pre-empt it by listing a maybe-alternative here. Ask "
        "ONLY ASK WHAT CAN CHANGE THE ANSWER: before asking anything, check the scheme's BASIC "
        "requirements against what the citizen has already told you. If an already-stated fact "
        "kills the scheme, it is closed - say so and move to a real alternative; do NOT ask about "
        "its finer sub-conditions (marks, exam scores, institution type, course name), because no "
        "answer to those can revive it. Found live (2026-09-02): told 'mere bete ne abhi 12th pass "
        "ki hai', the reply correctly stated the scheme was for students CURRENTLY STUDYING in "
        "class 10 or 12 - and then, in that same reply, asked what percentage he scored in 12th and "
        "whether his school was government or private. Both facts were irrelevant the moment 'has "
        "already passed' was known; the citizen answered '70%' for nothing and got the same "
        "rejection a turn later, on the exact criterion already visible in the first message. Work "
        "in this order: does a stated fact already rule this scheme out (close it), then is a fact "
        "that DECIDES eligibility still unknown (ask that), and only then anything finer. A "
        "question that leaves you rejecting the scheme for the same reason regardless of the answer "
        "must not be asked at all. "
        "Ask "
        "at most 2 questions per reply, each about ONE thing, never combined. Prefer broad, "
        "low-sensitivity questions (age, class/degree, income, residence) over narrow or sensitive "
        "ones (disability, caste, marital status, gender) - ask about the latter only once no broad "
        "fact remains unconfirmed. Prefer answering (one named scheme, no question attached) over "
        "asking again whenever you already have enough for a reasonable recommendation - only keep "
        "asking if the missing fact would change WHICH scheme applies or whether they're eligible at "
        "all, not for minor refinements. A 'minor refinement' means something that affects HOW they "
        "apply or how much they get - a document to arrange, a deadline, a step at the office - "
        "never a condition that decides IF they qualify. Any threshold stated in the scheme's own "
        "eligibility text (a land size, an income limit, an age cutoff, a family size, a category) "
        "decides IF, so it is never a minor refinement, however small the number sounds - if the "
        "citizen's value for it isn't confirmed, that is the question to ask. Caveats are for the "
        "genuinely minor kind ONLY: writing an IF-condition as a caveat is not a way to avoid "
        "asking about it, it is the same omission with different punctuation, and it pushes work "
        "onto a citizen who came here precisely because they can't read eligibility rules "
        "themselves. Use the full 'Scheme text' once you're naming that one scheme with no question "
        "attached - 'Scheme summaries' exist for other internal uses, not for padding this answer "
        "with a second, unconfirmed scheme.\n\n"
        f"{hard_stop_block}"
        f"{_RECENCY_RULE}\n\n"
        f"{_format_conversation_history(state.get('conversation_history'))}"
        f"{_format_profile(state.get('profile'))}"
        f"{_format_notes(state.get('conversation_notes'))}"
        f"{question_block}\n\n"
        f"Structured eligibility data:\n{structured_text}\n\n"
        f"Scheme summaries (short - use these instead of 'Scheme text' when asking a clarifying "
        f"question):\n{discovery_cards_text}\n\n"
        f"Scheme text (full detail - use once no clarifying question is needed):\n{chunks_text}\n\n"
        f"{lang_instruction} Keep it concise and practical."
    )
    result = _invoke_structured(get_reply_llm(), SchemeAnswer, [("system", PERSONA), ("human", prompt)], label="generate")
    # Mechanically enforced retry (2026-08-19) - not just another prompt
    # wording tweak. The "don't name a scheme in the same reply as a
    # question" rule above has now been violated THREE separate times in a
    # row across live testing (leading with names before any question,
    # naming alongside a still-live question, naming alongside a
    # conditional "try this if X" hedge), despite the prompt instruction
    # getting progressively more explicit each time. Same reasoning as
    # generate_clarify_rounds' HARD STOP below: once a rule needs to be a
    # GUARANTEE rather than a strong hint, keep patching the wording stops
    # being the right lever - check the actual output and correct it
    # instead. One retry, with the SPECIFIC violation named back to the
    # model (which scheme name(s) leaked into a question-asking reply),
    # rather than a generic re-statement of the same rule it just missed.
    scheme_names = list(structured_by_scheme.keys())
    leaked_names = [n for n in scheme_names if n.lower() in result.reply.lower()]
    if result.asked_clarifying_question and leaked_names:
        correction_prompt = (
            f"{prompt}\n\n"
            f"Your previous answer named a scheme ({', '.join(leaked_names)}) AND asked a "
            "clarifying question in the same reply - not allowed. Rewrite as ONLY ONE of these: "
            "either ask the clarifying question again with ZERO scheme names anywhere in the "
            "reply, or give the complete final answer naming that ONE scheme with NO question "
            "attached at all.\n"
            f"{_REWRITE_FIELD_REMINDER}"
        )
        result = _invoke_structured(
            get_reply_llm(), SchemeAnswer, [("system", PERSONA), ("human", correction_prompt)], label="generate_fix"
        )
    # Second, separate mechanical retry (2026-09-01) for the mirror-image
    # violation: naming a scheme with NO question attached (asked_
    # clarifying_question=False) but a hedge/caveat instead ("fits you - but
    # if your land is under 1.5 hectare it won't, let me know"). The prompt's
    # "HAVING THE CONVERSATION" rule above already says this doesn't count as
    # a complete answer, but that's a hint, not a guarantee - same reasoning
    # as the leaked_names retry just above and generate_clarify_rounds' HARD
    # STOP further down: once a rule has to be a GUARANTEE, check the actual
    # output and correct it rather than trust the wording alone. Found live
    # (2026-09-01): a farmer who'd only confirmed owning land (not its size)
    # was told Kantedar Tarbandi Yojana fit him, hedged with "agar 1.5
    # hectare se kam hai to fit nahi karega, bata dena" - the model's own
    # self-report (named_scheme_has_unconfirmed_condition) catches exactly
    # this pattern even though it isn't a bare scheme-name-leak like the
    # check above, since asked_clarifying_question was already False there.
    #
    # Gated on generate_clarify_rounds < 2 - WITHOUT that guard this retry
    # would fight the HARD STOP directly below it: on a turn where the cap
    # has already been reached, the prompt tells the model to stop asking and
    # commit to a best-effort answer (necessarily caveated, since the facts
    # never arrived), which is exactly the shape this retry rewrites BACK
    # into a question - re-opening the seven-consecutive-questions bug
    # generate_clarify_rounds exists to prevent. Past the cap, a caveated
    # answer is the intended outcome, not a violation.
    elif (
        not result.asked_clarifying_question
        and result.named_scheme_has_unconfirmed_condition
        and generate_clarify_rounds < 2
    ):
        correction_prompt = (
            f"{prompt}\n\n"
            "Your previous answer named a scheme but left a condition in its eligibility text "
            "unconfirmed (a hedge/caveat like 'if your land is under X, this won't fit - let me "
            "know' instead of an actual question). Rewrite as ONLY the clarifying question needed "
            "to confirm that specific condition - ZERO scheme names anywhere in the reply. Do not "
            "name the scheme again until that fact is actually confirmed by the citizen.\n"
            f"{_REWRITE_FIELD_REMINDER}"
        )
        result = _invoke_structured(
            get_reply_llm(), SchemeAnswer, [("system", PERSONA), ("human", correction_prompt)], label="generate_fix_hedge"
        )
    # Mechanically enforced regardless of what the model decided - the
    # HARD STOP prompt instruction above asks nicely, but prompt-following
    # isn't guaranteed, and the whole point of this cap is a GUARANTEE the
    # loop terminates, not just a strong hint. Once generate_clarify_rounds
    # already hit the cap coming into this turn, this turn is forced to
    # count as a final answer no matter what asked_clarifying_question says.
    asked_question = result.asked_clarifying_question and generate_clarify_rounds < 2
    # Set the SAME way clarify_node/clarify_identity_node already do, so a
    # terse follow-up reply ("nahi"/"18 saal") next turn has something
    # structured to anchor to - found live (2026-08-16): generate_node's own
    # clarifying questions had NO such anchor (only clarify_node/
    # clarify_identity_node set this), so a citizen's direct "nahi" to a
    # disability question never made it into subject_facts at all, and the
    # same question got asked again two turns later despite already being
    # answered. Cleared (None) when this reply didn't ask anything, so a
    # later unrelated message doesn't get wrongly treated as answering a
    # question that was never actually asked.
    return {
        "last_reply": result.reply,
        "pending_need_text": state.get("need_text") if asked_question else None,
        "pending_need_domain": "scheme" if asked_question else None,
        # Same pinning as clarify_identity_node (see ConvState.
        # pending_clarify_scheme_ids). This node asks its OWN clarifying
        # questions too - and in the live incident that prompted this it was
        # THIS node asking, about CM Anuprati Coaching Scheme's caste/income
        # criteria - so it has to pin just as clarify_identity_node does, or
        # answering it can evict the scheme it was asked for. Pinned to every
        # scheme whose text was actually in context for this reply, since
        # unlike clarify_identity_node there is no single `top` here and the
        # question can legitimately be about any of them. Cleared when the
        # reply asks nothing, so a pin can't linger onto a later, unrelated
        # turn.
        "pending_clarify_scheme_ids": (
            [meta["scheme_id"] for meta in structured_by_scheme.values()] if asked_question else None
        ),
        "generate_clarify_rounds": (generate_clarify_rounds + 1) if asked_question else 0,
    }


def chitchat_node(state: ConvState):
    """Fallback only - the normal path composes the chitchat reply directly
    inside extract_need_node (NeedExtraction.chitchat_reply) and skips this
    node entirely via route_after_extract. This only runs if that field
    came back empty despite intent='chitchat'."""
    lang_instruction = _lang_instruction(state.get("reply_lang"))
    prompt = (
        f'The citizen said: "{state["normalized_message"]}"\n'
        f"This isn't a scheme/job question. Reply briefly and warmly. {lang_instruction} "
        "Mention you can help them find Rajasthan government schemes and jobs."
    )
    reply = _invoke_reply(get_reply_llm(), [("system", PERSONA), ("human", prompt)], state.get("reply_lang"), label="chitchat")
    return {"last_reply": reply}


# ---------------------------------------------------------------------------
# Job-vacancy domain (parallel to the scheme Stage A/B/generate nodes above -
# see app/services/job_service.py's docstring for why this is a separate path rather
# than routed through RagService's eligibility-gating machinery)
# ---------------------------------------------------------------------------

def stage_a_jobs_node(state: ConvState):
    primary_need = (state.get("needs") or [None])[0]
    need_text = primary_need["need_text"] if primary_need else state.get("need_text", "")
    location = primary_need.get("job_location") if primary_need else None
    qualification = primary_need.get("job_qualification") if primary_need else None
    min_salary = primary_need.get("job_min_salary") if primary_need else None
    turn_type = state.get("turn_type")

    svc = get_job_service()
    already_shown = list(state.get("already_shown_job_ids") or [])
    # "show other jobs" excludes what's already been shown; a fresh/different
    # search (even a repeat of the same title) should not - a citizen asking
    # about "Patwari" again later should still see it.
    exclude_ids = already_shown if turn_type == "wants_alternatives" else None
    results = svc.search(
        need_text, location=location, exclude_ids=exclude_ids, top_n=5,
        qualification=qualification, min_salary=min_salary,
    )

    shown_jobs = [{"job_id": j["id"], "title": j["title"], "department": j["department"]} for j in results]
    # Record EVERY job shown this turn, not just the first - a "show more"
    # excludes already_shown_job_ids, so if only the top result were tracked,
    # postings #2-5 from an earlier batch could reappear in a later "show
    # more" call (found testing this directly: "Police Constable" shown as
    # the #2 result in one turn reappeared in the very next "aur konsi jobs
    # hai" call, since only the #1 result's id had been recorded as shown).
    for j in results:
        if j["id"] not in already_shown:
            already_shown.append(j["id"])

    return {
        "job_shortlist": results,
        # `or state.get(...)` fallback (2026-08-25, found live) - the scheme
        # side's equivalent (show_more_node) already guards this way; this
        # node didn't, so a "show more jobs" search that comes back empty
        # (the remaining list is genuinely exhausted) used to silently wipe
        # out whatever job was already in focus, breaking a subsequent
        # "tell me more about that one" followup (route_after_extract's
        # shown_job_id check) into a fresh, unrelated search instead.
        "shown_jobs": shown_jobs or state.get("shown_jobs"),
        "shown_job_id": results[0]["id"] if results else state.get("shown_job_id"),
        "already_shown_job_ids": already_shown,
        # For route_after_stage_a_jobs - see JobService.has_title_match's
        # docstring for why this needs its own signal, distinct from
        # job_location/job_qualification/job_min_salary.
        "job_title_named": svc.has_title_match(need_text),
    }


def stage_b_followup_jobs_node(state: ConvState):
    shown_jobs = state.get("shown_jobs") or []
    target_title = state.get("followup_target")
    targets = [j for j in shown_jobs if j["title"] == target_title] if target_title else []
    if not targets:
        targets = shown_jobs or ([{"job_id": state["shown_job_id"]}] if state.get("shown_job_id") else [])

    svc = get_job_service()
    jobs = [svc.get(t["job_id"]) for t in targets]
    jobs = [j for j in jobs if j]
    return {"job_shortlist": jobs}


_JOB_FIELD_FOCUS_BY_INTENT = {
    "how_to_apply": "focus on the official application portal link and what documents are needed to apply.",
    "documents_question": "focus ONLY on exactly what documents are needed.",
    "ask_detail": "focus on qualification required, salary, vacancies, and the application deadline.",
}


def generate_jobs_node(state: ConvState):
    lang_instruction = _lang_instruction(state.get("reply_lang"))
    jobs = state.get("job_shortlist") or []

    if not jobs:
        prompt = (
            f'The citizen asked about a government job: "{state.get("need_text", "")}"\n'
            f"No matching postings were found. Say so briefly {lang_instruction} and suggest they "
            "check back later or ask about a different post/location. One or two sentences."
        )
        reply = _invoke_reply(get_reply_llm(), [("system", PERSONA), ("human", prompt)], state.get("reply_lang"), label="generate_jobs_empty")
        return {"last_reply": reply}

    jobs_desc = "\n".join(
        f"- {j['title']} ({j['department']}), location: {j['location']}, status: {j['status']}, "
        f"vacancies: {j['vacancies']}, qualification required: {j['qualification']}, "
        f"salary: {j['salary']}, applications open {j['startDate']} to {j['lastDate']}, "
        f"documents needed: {', '.join(j['documentsNeeded'])}, "
        f"application portal: {j['officialApplicationPortal']}"
        for j in jobs
    )
    focus = _JOB_FIELD_FOCUS_BY_INTENT.get(state.get("intent"), "give a clear overview of the matching postings.")

    # The citizen's OWN structured qualification/location/salary (2026-08-19,
    # found live missing here) - previously this prompt only had need_text/
    # raw_message free text to work with, no structured signal at all, even
    # though NeedItem.job_qualification is already extracted into a clean
    # closed set (12th Pass/Diploma/Graduate/Post Graduate). Found live:
    # "maine ITI aur B.COM kiya hua hai" (a COMPOUND qualification - ITI is
    # Diploma-tier, B.Com is Graduate-tier) got reasoned about from raw text
    # alone and mishandled. Passing the already-extracted value directly
    # gives the model a clean anchor instead of re-deriving it from scratch
    # each time from a sentence that states two qualifications at once.
    primary_need = (state.get("needs") or [None])[0] or {}
    known_bits = [
        f"{label}: {primary_need[field]}"
        for field, label in (("job_qualification", "qualification"), ("job_location", "preferred location"), ("job_min_salary", "minimum salary wanted"))
        if primary_need.get(field)
    ]
    known_block = f"What's already confirmed about the citizen: {'; '.join(known_bits)}\n" if known_bits else ""

    # Under-qualification is decided in CODE, not left to the model to notice
    # from the postings list (2026-09-02, found live). search() is
    # deliberately fail-open on qualification - correct for OVER-qualification
    # (a Graduate may still apply to a 12th-Pass post) but wrong for UNDER-
    # qualification, where the citizen cannot apply to anything here at all.
    # A 10th-pass citizen was shown five 12th-Pass-only vacancies laid out as
    # normal options, with the mismatch mentioned only in passing above them
    # and a repeat of the qualification question below them - the citizen has
    # to work out for themselves that none of it applies. If nothing in the
    # corpus is reachable for them, the reply must lead with that plainly and
    # not present the listings as things to apply for.
    under_qualified = is_under_qualified_for_all(primary_need.get("job_qualification"))
    if under_qualified:
        known_block += (
            "IMPORTANT - the citizen's stated qualification is BELOW the minimum requirement of "
            "EVERY posting in this dataset (they all require at least 12th Pass). Lead with that "
            "plainly and kindly in your first sentence: right now there is no vacancy here they "
            "can apply for. Do NOT list the postings below as options for them, do NOT format "
            "them as an apply-to list, and do NOT re-ask what their qualification is - they have "
            "already told you. You may briefly note what the lowest requirement is (12th Pass) so "
            "they know what would open these up, and offer to check again if they complete it. "
            "Keep the whole reply short - a list they cannot use is not helpful to them.\n"
        )

    # Conversation context (2026-08-18): job replies used to be composed
    # from the distilled need_text alone, blind to the conversation - a
    # followup or a short answer to the bot's own question got answered as
    # if it were the first message of a brand-new session.
    prompt = (
        f"Today's date is {date.today().isoformat()}.\n"
        f"{_RECENCY_RULE}\n\n"
        f"{_format_conversation_history(state.get('conversation_history'))}"
        f"{_format_profile(state.get('profile'))}"
        f"{_format_notes(state.get('conversation_notes'))}"
        f"{known_block}"
        f'The citizen asked: "{state.get("need_text", "")}"\n'
        f'Citizen\'s exact words this turn: "{state.get("raw_message", "")}"\n'
        f"Matching government job posting(s):\n{jobs_desc}\n\n"
        f"Answer their question {lang_instruction} using ONLY this data - never invent a detail "
        f"not listed above. {focus} Use today's date above to reason correctly about whether the "
        "application window has already started, is upcoming, or has already closed - do not guess "
        "this from the dates alone without comparing them to today. The 'status' field is the "
        "authoritative signal for whether it's open ('active'), not yet open ('coming soon'), or "
        "over ('closed') - if your date reasoning and the status field seem to disagree, trust "
        "status and phrase the reply around it rather than asserting a contradiction. Format as a "
        "short list if there are multiple postings.\n\n"
        "Do not call any single posting the 'best'/'sabse best' option unless you can point to a "
        "SPECIFIC stated preference (location, salary, deadline, qualification fit) that makes it "
        "clearly the strongest match - if you don't have that, present them as ranked matches with "
        "the reasoning shown ('yeh sabse zyada fit baithti hai kyunki...'), not an unqualified "
        "absolute claim. If nothing here is currently active/open, say so plainly and offer to "
        "notify them when a matching posting opens, rather than recommending a closed one as if it "
        "were usable now."
    )
    reply = _invoke_reply(get_reply_llm(), [("system", PERSONA), ("human", prompt)], state.get("reply_lang"), label="generate_jobs")
    return {"last_reply": reply}


_NO_MATCH_TEXT = {
    "hindi": (
        "माफ़ कीजिए, मुझे आपके लिए कोई योजना नहीं मिली। कृपया थोड़ा और बताइए कि आपको किस चीज़ के लिए "
        "मदद चाहिए।"
    ),
    "hinglish": (
        "Maaf kijiye, mujhe aapke liye koi scheme nahi mili. Kripya thoda aur bataiye "
        "ki aapko kis cheez ke liye madad chahiye."
    ),
    "english": "Sorry, I couldn't find a matching scheme. Could you tell me a bit more about "
    "what kind of help you need?",
}


def no_match_node(state: ConvState):
    text = _NO_MATCH_TEXT.get(state.get("reply_lang"), _NO_MATCH_TEXT["english"])
    return {"last_reply": text}


# ---------------------------------------------------------------------------
# Jan Aadhaar citizen-status verification sub-flow (mock, 2026-08-14) - a
# scripted 3-step OTP-consent dialogue (member choice -> OTP -> fetch),
# mirroring the real Jan Aadhaar Integration Document v1.8 flow. Every
# message in it is parsed deterministically via app/utils/pii_utils.py, never by
# an LLM - see route_after_normalize/route_after_extract above for how turns
# get routed here instead of through the normal extraction pipeline.
# ---------------------------------------------------------------------------

_ASK_CONSENT_TEXT = {
    "hindi": "आपकी स्थिति जानने के लिए मुझे आपका Jan Aadhaar verify करना होगा - इसके लिए 10 अंकों का Jan Aadhaar ID और एक OTP माँगना होगा। क्या आप इसके लिए सहमत हैं? (हाँ/नहीं)",
    "hinglish": "Aapki status jaanne ke liye mujhe aapka Jan Aadhaar verify karna hoga - iske liye 10 anko ka Jan Aadhaar ID aur ek OTP maangna hoga. Kya aap iske liye sahmat hain? (Haan/Nahi)",
    "english": "To check your status, I'll need to verify your Jan Aadhaar - that means asking for your 10-digit Jan Aadhaar ID and an OTP. Do you consent to this? (Yes/No)",
}
_CONSENT_UNCLEAR_TEXT = {
    "hindi": "माफ़ कीजिए, मुझे समझ नहीं आया। क्या आप Jan Aadhaar verify करने के लिए सहमत हैं? कृपया हाँ या नहीं में जवाब दें।",
    "hinglish": "Maaf kijiye, samajh nahi aaya. Kya aap Jan Aadhaar verify karne ke liye sahmat hain? Kripya Haan ya Nahi mein jawab dein.",
    "english": "Sorry, I didn't quite catch that. Do you consent to verifying your Jan Aadhaar? Please reply Yes or No.",
}
_CONSENT_DECLINED_TEXT = {
    "hindi": "ठीक है, कोई बात नहीं। बिना Jan Aadhaar verify किए मैं यह जानकारी नहीं दे सकता, लेकिन मैं आपकी किसी और चीज़ में मदद कर सकता हूँ।",
    "hinglish": "Theek hai, koi baat nahi. Bina Jan Aadhaar verify kiye main yeh jaankari nahi de sakta, lekin main aapki kisi aur cheez mein madad kar sakta hoon.",
    "english": "No problem at all. I can't share this specific information without verifying your Jan Aadhaar, but I'm happy to help with anything else.",
}
_ASK_ID_TEXT = {
    "hindi": "कृपया अपना 10 अंकों का Jan Aadhaar ID बताएं।",
    "hinglish": "Kripya apna 10 anko ka Jan Aadhaar ID bataiye.",
    "english": "Please share your 10-digit Jan Aadhaar ID.",
}
_ID_INVALID_TEXT = {
    "hindi": "माफ़ कीजिए, यह एक मान्य Jan Aadhaar ID नहीं लग रहा। कृपया सही 10 अंकों का ID दोबारा भेजें।",
    "hinglish": "Maaf kijiye, yeh valid Jan Aadhaar ID nahi lag raha. Kripya sahi 10 anko ka ID dobara bhejein.",
    "english": "Sorry, that doesn't look like a valid Jan Aadhaar ID. Please send the correct 10-digit ID again.",
}
_ID_NOT_FOUND_TEXT = {
    "hindi": "माफ़ कीजिए, इस Jan Aadhaar ID से कोई रिकॉर्ड नहीं मिला। कृपया ID दोबारा जांच कर बताएं, या कुछ और पूछें।",
    "hinglish": "Maaf kijiye, is Jan Aadhaar ID se koi record nahi mila. Kripya ID dobara check karke bataiye, ya kuch aur pooch sakte hain.",
    "english": "Sorry, no record was found for that Jan Aadhaar ID. Please double-check and try again, or ask me something else.",
}
_MEMBER_CHOICE_HEADER = {
    "hindi": "आपके Jan Aadhaar में ये सदस्य मिले हैं। कृपया बताएं यह जानकारी किसके लिए चाहिए (नंबर बताएं):",
    "hinglish": "Aapke Jan Aadhaar mein ye members mile hain. Kripya bataiye yeh jaankari kiske liye chahiye (number bataiye):",
    "english": "Your Jan Aadhaar has these family members - please tell me which one this is for (reply with the number):",
}
# Appended after the member list every time (2026-08-14) - a relative living
# in a DIFFERENT household has their own separate Jan Aadhaar family ID (per
# janaadhaar.rajasthan.gov.in's own FAQ: a split household gets "a new Jan
# Aadhaar family ID"), so they simply won't appear in THIS list at all - the
# member list itself carries no relation metadata (just masked names), so
# only the citizen can recognize whether the person they mean is even here.
# Offered proactively rather than only after a failed match, so a citizen
# asking about someone outside their own household doesn't have to guess
# their way through a wrong list first.
_MEMBER_CHOICE_FOOTER = {
    "hindi": "(अगर इनमें से कोई नहीं है, तो बताएं और उनका अपना Jan Aadhaar ID दें।)",
    "hinglish": "(Agar inme se koi nahi hai, to bataiye aur unka apna Jan Aadhaar ID dijiye.)",
    "english": "(If none of these are who you mean, let me know and share their own Jan Aadhaar ID instead.)",
}
_NOT_IN_LIST_TEXT = {
    "hindi": "ठीक है। अगर वह व्यक्ति आपके साथ एक ही Jan Aadhaar परिवार में नहीं हैं, तो कृपया उनका अपना 10 अंकों का Jan Aadhaar ID बताएं।",
    "hinglish": "Theek hai. Agar wo vyakti aapke saath ek hi Jan Aadhaar parivar mein nahi hain, to kripya unka apna 10 anko ka Jan Aadhaar ID bataiye.",
    "english": "Understood. If they're not part of your own Jan Aadhaar household, please share their own 10-digit Jan Aadhaar ID instead.",
}
_NONE_OF_THESE_PHRASES = (
    "inme se koi nahi", "in mein se koi nahi", "koi nahi hai", "koi bhi nahi",
    "इनमें से कोई नहीं", "कोई नहीं है", "none of these", "not in the list",
    "not on this list", "alag rehte", "अलग रहते",
)
_ASK_OTP_TEXT = {
    "hindi": "आपके रजिस्टर्ड मोबाइल नंबर पर एक OTP भेज दिया गया है। कृपया वह 6 अंकों का OTP यहां बताएं।",
    "hinglish": "Aapke registered mobile number par ek OTP bhej diya gaya hai. Kripya wo 6 anko ka OTP yahan bataiye.",
    "english": "An OTP has been sent to your registered mobile number. Please share that 6-digit OTP here.",
}
_MOCK_SMS_LABEL = {
    "hindi": "जन आधार से SMS",
    "hinglish": "Jan Aadhaar se SMS",
    "english": "SMS from Jan Aadhaar",
}


def _mock_sms_text(reply_lang, otp):
    """Mock-only (2026-08-17): builds the popup text the frontend shows to
    simulate an incoming SMS, since our own backend generates the OTP in
    mock mode and would otherwise only ever print it to server logs."""
    return f"📱 {_pick_text(reply_lang, _MOCK_SMS_LABEL)}: {otp}"
_OTP_INVALID_TEXT = {
    "hindi": "यह OTP सही नहीं लग रहा। कृपया दोबारा सही 6 अंकों का OTP भेजें।",
    "hinglish": "Yeh OTP sahi nahi lag raha. Kripya dobara sahi 6 anko ka OTP bhejein.",
    "english": "That OTP doesn't look correct. Please send the correct 6-digit OTP again.",
}
_FLOW_ABORTED_TEXT = {
    "hindi": "माफ़ कीजिए, वेरिफिकेशन पूरा नहीं हो पाया। कृपया थोड़ी देर बाद दोबारा कोशिश करें, या मुझसे कुछ और पूछें।",
    "hinglish": "Maaf kijiye, verification poora nahi ho paaya. Kripya thodi der baad dobara koshish karein, ya mujhse kuch aur poochein.",
    "english": "Sorry, verification couldn't be completed. Please try again in a little while, or ask me something else.",
}
# 2026-08-26 - see janaadhaar_handle_node's awaiting_otp handling: a resend
# request used to fall through to _OTP_INVALID_TEXT (a "that OTP is wrong"
# message, wrong for someone who never got one) AND burn a real attempt.
_OTP_RESENT_TEXT = {
    "hindi": "ठीक है, आपका OTP दोबारा भेज दिया गया है। कृपया वह OTP यहाँ भेजें।",
    "hinglish": "Theek hai, aapka OTP dobara bhej diya gaya hai. Kripya wahi OTP yahan bhejein.",
    "english": "Okay, your OTP has been sent again. Please enter it here.",
}
_RESEND_LIMIT_TEXT = {
    "hindi": "माफ़ कीजिए, OTP कई बार भेजा जा चुका है। कृपया अपना फ़ोन जांचें, या थोड़ी देर बाद दोबारा कोशिश करें।",
    "hinglish": "Maaf kijiye, OTP kai baar bhej diya gaya hai. Kripya apna phone jaanchein, ya thodi der baad dobara koshish karein.",
    "english": "Sorry, the OTP has already been resent several times. Please check your phone, or try again in a little while.",
}

_MEMBER_ORDINAL_WORDS = {
    "1": 0, "first": 0, "pehla": 0, "pehli": 0, "पहला": 0, "पहली": 0,
    "2": 1, "second": 1, "dusra": 1, "dusri": 1, "दूसरा": 1, "दूसरी": 1,
    "3": 2, "third": 2, "teesra": 2, "teesri": 2, "तीसरा": 2, "तीसरी": 2,
}

_MAX_JANAADHAAR_ATTEMPTS = 3
# Separate, smaller cap on OTP resends specifically - own limit, not shared
# with _MAX_JANAADHAAR_ATTEMPTS above, since a resend request isn't a
# failed guess (see janaadhaar_resend_count's docstring in app/graph/state.py).
_MAX_JANAADHAAR_RESENDS = 2


def _pick_text(reply_lang, texts):
    return texts.get(reply_lang, texts["english"])


def _format_member_choice_text(reply_lang, members):
    header = _pick_text(reply_lang, _MEMBER_CHOICE_HEADER)
    lines = "\n".join(f"{i + 1}. {m['masked_name']}" for i, m in enumerate(members))
    footer = _pick_text(reply_lang, _MEMBER_CHOICE_FOOTER)
    return f"{header}\n{lines}\n{footer}"


def _is_none_of_these(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in _NONE_OF_THESE_PHRASES)


def _resolve_member_choice(text, members):
    # Tokenize rather than requiring the WHOLE trimmed message to match one
    # of the ordinal words exactly - found live (2026-08-14): "dusra wala"
    # ("the second one", a natural Hinglish phrasing with a trailing "wala")
    # failed an exact-match lookup even though "dusra" itself is a known
    # ordinal, because the full string "dusra wala" isn't itself a dict key.
    words = re.findall(r"[\wऀ-ॿ]+", text.lower())
    for word in words:
        if word in _MEMBER_ORDINAL_WORDS:
            idx = _MEMBER_ORDINAL_WORDS[word]
            if idx < len(members):
                return members[idx]
    digits = pii_utils.digits_only(text)
    if digits:
        idx = int(digits) - 1
        if 0 <= idx < len(members):
            return members[idx]
    return None


def _abort_janaadhaar_flow(reply_lang):
    return {
        "last_reply": _pick_text(reply_lang, _FLOW_ABORTED_TEXT),
        "pending_janaadhaar_step": None,
        "janaadhaar_member_id": None,
        "janaadhaar_tid": None,
        "janaadhaar_members": [],
        "janaadhaar_attempts": 0,
    }


def _answer_from_entitlements(need_text, entitlements, reply_lang):
    """Shared answer-composition prompt, used both right after a fresh OTP
    success and (2026-08-18) for a same-session followup that reuses already-
    cached entitlement data instead of re-verifying - see
    janaadhaar_handle_node's `step is None` branch."""
    lang_instruction = _lang_instruction(reply_lang)
    prompt = (
        f'The citizen asked about their own status: "{need_text}"\n'
        f"Their identity has just been verified via Jan Aadhaar OTP. Their scheme entitlements:\n"
        f"{entitlements}\n\n"
        f"Answer their question directly using ONLY this data {lang_instruction} - match it to "
        "whichever scheme they actually asked about (by name, even if phrased differently, "
        "abbreviated, or in Hindi/Hinglish). If they asked generally what they're enrolled in "
        "rather than about one specific scheme, briefly list all of the entitlements above. If "
        "NONE of the entitlements match what they asked about, say plainly that no record was "
        "found for that specific scheme, rather than guessing from an unrelated one. If the "
        "entitlement itself IS found but doesn't contain the SPECIFIC detail they're asking about "
        "(e.g. an exact date/timeline for when a pending verification will complete - this data "
        "only has a status label, never a completion ETA), say plainly that this specific detail "
        "isn't available, and share whatever status information you do have instead of guessing at "
        "or inventing a timeline. Do not invent any detail not listed above. Keep it to one or two "
        "sentences unless listing multiple entitlements. Do not mention Jan Aadhaar, OTP, or the "
        "verification process in this reply - just answer their status question naturally, as if "
        "you already knew this about them."
    )
    return _invoke_reply(get_reply_llm(), [("system", PERSONA), ("human", prompt)], reply_lang, label="answer_from_entitlements")


def janaadhaar_handle_node(state: ConvState):
    step = state.get("pending_janaadhaar_step")
    reply_lang = state.get("reply_lang")
    attempts = state.get("janaadhaar_attempts", 0)
    svc = get_janaadhaar_service()

    if step is None:
        # Same-session followup reusing already-verified data (2026-08-18):
        # found live - "palanhar kab tak verify hojayga" right after a
        # successful verification restarted the WHOLE ID/OTP flow from
        # scratch, even though janaadhaar_status_data already held that
        # citizen's fetched entitlements. state.py's own docstring for that
        # field always documented this intent ("kept in case a same-session
        # followup asks about it again without re-verifying") but nothing
        # ever actually checked it before this fix - re-verifying was the
        # ONLY path. Answer straight from the cache instead; only fall
        # through to asking for the ID when there's nothing cached yet.
        cached = state.get("janaadhaar_status_data")
        if cached:
            # Only reuse the cache for the SAME person it was fetched for
            # (2026-08-25, found live) - without this check, a status
            # question about a DIFFERENT family member than the one
            # originally verified silently answered using the wrong
            # person's cached entitlements, with no re-verification and no
            # indication anything was off.
            primary_need = (state.get("needs") or [None])[0]
            current_subject = ((primary_need.get("subject_relation") if primary_need else None) or "self").strip().lower()
            cached_subject = (state.get("janaadhaar_status_data_for") or "self").strip().lower()
            if current_subject == cached_subject:
                reply = _answer_from_entitlements(state.get("need_text", ""), cached, reply_lang)
                return {"last_reply": reply}
            # Different person - fall through to a fresh verification rather
            # than answering from the stale cache.
        # Fresh trigger from a citizen_status need - THIS turn's own message
        # (the status question itself) already safely went through
        # extract_profile/extract_need, nothing to redact yet. Ask for
        # explicit consent BEFORE requesting the ID/OTP at all (found live
        # 2026-08-19, tester-reported gap: this flow used to go straight to
        # asking for the ID with no consent step whatsoever).
        return {
            "last_reply": _pick_text(reply_lang, _ASK_CONSENT_TEXT),
            "pending_janaadhaar_step": "awaiting_consent",
            "janaadhaar_attempts": 0,
        }

    # From here on, THIS turn's raw_message is itself a credential attempt
    # (a consent answer, an ID, a member choice, or an OTP) - never let it
    # reach conversation_history/logs unredacted, regardless of which branch
    # below actually runs. Consent itself isn't sensitive, but redacting it
    # too is the safe default - a confused citizen could easily blurt out
    # their actual ID in response to the consent question instead of a
    # plain yes/no, and this net catches that either way.
    redacted = {"redacted_raw_message": "[Jan Aadhaar verification input redacted]"}

    if step == "awaiting_consent":
        consent = pii_utils.extract_consent(state["raw_message"])
        if consent is False:
            return {**redacted, **_abort_janaadhaar_flow(reply_lang), "last_reply": _pick_text(reply_lang, _CONSENT_DECLINED_TEXT)}
        if consent is None:
            if attempts + 1 >= _MAX_JANAADHAAR_ATTEMPTS:
                return {**redacted, **_abort_janaadhaar_flow(reply_lang)}
            return {**redacted, "last_reply": _pick_text(reply_lang, _CONSENT_UNCLEAR_TEXT), "janaadhaar_attempts": attempts + 1}
        return {
            **redacted,
            "last_reply": _pick_text(reply_lang, _ASK_ID_TEXT),
            "pending_janaadhaar_step": "awaiting_id",
            "janaadhaar_attempts": 0,
        }

    if step == "awaiting_id":
        jan_id = pii_utils.extract_jan_aadhaar_id(state["raw_message"])
        if not jan_id:
            if attempts + 1 >= _MAX_JANAADHAAR_ATTEMPTS:
                return {**redacted, **_abort_janaadhaar_flow(reply_lang)}
            return {**redacted, "last_reply": _pick_text(reply_lang, _ID_INVALID_TEXT), "janaadhaar_attempts": attempts + 1}

        result = svc.fetch_member_list(jan_id)
        if result["status"] != "JAN_200" or not result["members"]:
            return {**redacted, **_abort_janaadhaar_flow(reply_lang)}

        members = result["members"]
        if len(members) == 1:
            otp_result = svc.generate_otp(members[0]["member_id"])
            return {
                **redacted,
                "last_reply": _pick_text(reply_lang, _ASK_OTP_TEXT),
                "pending_janaadhaar_step": "awaiting_otp",
                "janaadhaar_member_id": members[0]["member_id"],
                "janaadhaar_tid": otp_result["tid"],
                "janaadhaar_members": members,
                "janaadhaar_attempts": 0,
                "janaadhaar_resend_count": 0,
                "mock_sms_preview": (
                    _mock_sms_text(reply_lang, otp_result["mock_otp"]) if otp_result.get("mock_otp") else None
                ),
            }
        return {
            **redacted,
            "last_reply": _format_member_choice_text(reply_lang, members),
            "pending_janaadhaar_step": "awaiting_member_choice",
            "janaadhaar_members": members,
            "janaadhaar_attempts": 0,
        }

    if step == "awaiting_member_choice":
        members = state.get("janaadhaar_members") or []
        if _is_none_of_these(state["raw_message"]):
            # The person they mean isn't in THIS family's list at all - most
            # likely a relative in a different household (their own,
            # separate Jan Aadhaar family ID) rather than a resolution
            # failure - restart the flow asking for THAT ID instead of
            # re-showing a list that can never contain them.
            return {
                **redacted,
                "last_reply": _pick_text(reply_lang, _NOT_IN_LIST_TEXT),
                "pending_janaadhaar_step": "awaiting_id",
                "janaadhaar_members": [],
                "janaadhaar_attempts": 0,
            }
        chosen = _resolve_member_choice(state["raw_message"], members)
        if not chosen:
            if attempts + 1 >= _MAX_JANAADHAAR_ATTEMPTS:
                return {**redacted, **_abort_janaadhaar_flow(reply_lang)}
            return {
                **redacted,
                "last_reply": _format_member_choice_text(reply_lang, members),
                "janaadhaar_attempts": attempts + 1,
            }

        otp_result = svc.generate_otp(chosen["member_id"])
        return {
            **redacted,
            "last_reply": _pick_text(reply_lang, _ASK_OTP_TEXT),
            "pending_janaadhaar_step": "awaiting_otp",
            "janaadhaar_member_id": chosen["member_id"],
            "janaadhaar_tid": otp_result["tid"],
            "janaadhaar_attempts": 0,
            "janaadhaar_resend_count": 0,
            "mock_sms_preview": (
                _mock_sms_text(reply_lang, otp_result["mock_otp"]) if otp_result.get("mock_otp") else None
            ),
        }

    if step == "awaiting_otp":
        # Checked BEFORE treating a non-6-digit message as a wrong guess
        # (2026-08-26, found live) - "OTP nahi mila"/"dobara bhejo" used to
        # fall straight through to extract_otp() returning None, read
        # exactly like a wrong OTP: shown a misleading "that OTP doesn't
        # look correct" message, AND it silently burned one of the limited
        # janaadhaar_attempts - for a delivery problem, not the citizen's
        # mistake. Own separate cap (_MAX_JANAADHAAR_RESENDS), not shared
        # with janaadhaar_attempts, so it can't stall the flow indefinitely
        # either.
        if pii_utils.extract_resend_request(state["raw_message"]):
            resend_count = state.get("janaadhaar_resend_count", 0)
            if resend_count >= _MAX_JANAADHAAR_RESENDS:
                return {**redacted, "last_reply": _pick_text(reply_lang, _RESEND_LIMIT_TEXT)}
            otp_result = svc.generate_otp(state.get("janaadhaar_member_id"))
            if otp_result.get("status") != "JAN_200":
                return {**redacted, **_abort_janaadhaar_flow(reply_lang)}
            return {
                **redacted,
                "last_reply": _pick_text(reply_lang, _OTP_RESENT_TEXT),
                "janaadhaar_tid": otp_result["tid"],
                "janaadhaar_resend_count": resend_count + 1,
                "mock_sms_preview": (
                    _mock_sms_text(reply_lang, otp_result["mock_otp"]) if otp_result.get("mock_otp") else None
                ),
            }

        otp = pii_utils.extract_otp(state["raw_message"])
        if not otp:
            if attempts + 1 >= _MAX_JANAADHAAR_ATTEMPTS:
                return {**redacted, **_abort_janaadhaar_flow(reply_lang)}
            return {**redacted, "last_reply": _pick_text(reply_lang, _OTP_INVALID_TEXT), "janaadhaar_attempts": attempts + 1}

        result = svc.validate_otp_and_fetch(state.get("janaadhaar_member_id"), state.get("janaadhaar_tid"), otp)
        if result["status"] != "JAN_200":
            if attempts + 1 >= _MAX_JANAADHAAR_ATTEMPTS:
                return {**redacted, **_abort_janaadhaar_flow(reply_lang)}
            return {**redacted, "last_reply": _pick_text(reply_lang, _OTP_INVALID_TEXT), "janaadhaar_attempts": attempts + 1}

        data = result["data"]

        # Deliberately NOT merged into the shared `profile` used for scheme
        # eligibility gating elsewhere (found live 2026-08-17, not yet fixed):
        # the member just verified here isn't necessarily the citizen
        # themselves - fetch_member_list returns every member on the Jan
        # Aadhaar (the whole family), and the citizen can pick any one of
        # them to check status for (e.g. "kya meri maa ki pension aayi"). The
        # mock member records carry no relation-to-self field, so there's
        # currently no way to confirm the verified member IS the citizen -
        # merging blindly would silently misattribute a family member's
        # gender/age/marital_status etc. as the citizen's own for every
        # unrelated scheme question for the rest of the conversation. Only
        # used to answer THIS status question below, not persisted anywhere.

        # Entitlement Details is a SEPARATE fetch from the identity data
        # above (unverified schema - see janaadhaar_mock_data.
        # handle_entitlement_details) - this is what actually carries
        # per-scheme status (enrolled/approved/paid, last payment amount/
        # date), not the identity fetch itself.
        entitlement_result = svc.get_entitlement_details(state.get("janaadhaar_member_id"))
        entitlements = entitlement_result.get("entitlements", [])

        # Only entitlements (never the ID/OTP, which never reached this
        # point as anything but a local variable) go into this prompt.
        reply = _answer_from_entitlements(state.get("need_text", ""), entitlements, reply_lang)
        primary_need = (state.get("needs") or [None])[0]
        status_data_for = (primary_need.get("subject_relation") if primary_need else None) or "self"
        return {
            **redacted,
            "last_reply": reply,
            "janaadhaar_status_data": entitlements,
            "janaadhaar_status_data_for": status_data_for,
            "pending_janaadhaar_step": None,
            "janaadhaar_member_id": None,
            "janaadhaar_tid": None,
            "janaadhaar_members": [],
            "janaadhaar_attempts": 0,
        }

    # Defensive fallback - shouldn't be reachable given the three states above.
    return {**redacted, **_abort_janaadhaar_flow(reply_lang)}


def finalize_node(state: ConvState):
    """Every path (generate/clarify/clarify_identity/chitchat/no_match) routes
    through here before END, so conversation_history updates consistently
    regardless of which one produced the reply - rather than duplicating
    this update in five different terminal nodes."""
    history = list(state.get("conversation_history") or [])
    # redacted_raw_message (set by janaadhaar_handle_node) stands in for the
    # real message whenever this turn was a Jan Aadhaar ID/OTP attempt - this
    # history gets replayed into FUTURE extract_profile/extract_need/generate
    # prompts, so persisting the raw value here would leak it to an LLM one
    # turn later even if this turn's own handling never did.
    persisted_message = state.get("redacted_raw_message") or state["raw_message"]
    history.append({"user": persisted_message, "bot": state.get("last_reply", "")})
    return {"conversation_history": history[-MAX_HISTORY_TURNS:]}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(ConvState)
    graph.add_node("normalize", normalize_node)
    graph.add_node("extract_profile", extract_profile_node)
    graph.add_node("extract_need", extract_need_node)
    graph.add_node("stage_a", stage_a_node)
    graph.add_node("stage_b", stage_b_node)
    graph.add_node("stage_b_followup", stage_b_followup_node)
    graph.add_node("show_more", show_more_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("clarify_identity", clarify_identity_node)
    graph.add_node("chitchat", chitchat_node)
    graph.add_node("no_match", no_match_node)
    graph.add_node("generate", generate_node)
    graph.add_node("stage_a_jobs", stage_a_jobs_node)
    graph.add_node("clarify_jobs", clarify_jobs_node)
    graph.add_node("stage_b_followup_jobs", stage_b_followup_jobs_node)
    graph.add_node("generate_jobs", generate_jobs_node)
    graph.add_node("janaadhaar_handle", janaadhaar_handle_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("normalize")
    graph.add_conditional_edges(
        "normalize",
        route_after_normalize,
        {"janaadhaar_handle": "janaadhaar_handle", "extract_profile": "extract_profile"},
    )
    graph.add_edge("extract_profile", "extract_need")
    graph.add_conditional_edges(
        "extract_need",
        route_after_extract,
        {
            "stage_b_followup": "stage_b_followup",
            "show_more": "show_more",
            "chitchat": "chitchat",
            "finalize": "finalize",
            "stage_a": "stage_a",
            "stage_a_jobs": "stage_a_jobs",
            "stage_b_followup_jobs": "stage_b_followup_jobs",
            "janaadhaar_handle": "janaadhaar_handle",
        },
    )
    graph.add_conditional_edges(
        "stage_a",
        route_after_stage_a,
        {"no_match": "no_match", "clarify": "clarify", "clarify_identity": "clarify_identity", "stage_b": "stage_b"},
    )
    graph.add_edge("stage_b", "generate")
    graph.add_edge("stage_b_followup", "generate")
    graph.add_edge("show_more", "generate")
    graph.add_edge("generate", "finalize")
    graph.add_conditional_edges(
        "stage_a_jobs",
        route_after_stage_a_jobs,
        {"clarify_jobs": "clarify_jobs", "generate_jobs": "generate_jobs"},
    )
    graph.add_edge("stage_b_followup_jobs", "generate_jobs")
    graph.add_edge("generate_jobs", "finalize")
    graph.add_edge("clarify_jobs", "finalize")
    graph.add_edge("clarify", "finalize")
    graph.add_edge("clarify_identity", "finalize")
    graph.add_edge("chitchat", "finalize")
    graph.add_edge("no_match", "finalize")
    graph.add_edge("janaadhaar_handle", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=MemorySaver())
