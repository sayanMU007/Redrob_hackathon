"""
Redrob Intelligent Candidate Ranking System

Senior AI Engineer — Founding Team @ Redrob AI

Architecture: Multi-signal hybrid scoring with semantic profile understanding.

Scoring pipeline:
  1. Title / Role Alignment   (35%) — proximity to "AI Engineer" archetype
  2. Skills Match             (25%) — core AI/ML skills with proficiency & duration weighting
  3. Career Quality           (20%) — product co. experience, trajectory, no red flags
  4. Experience Calibration   (10%) — 5-9 yr sweet spot per JD
  5. Location / Logistics      (5%) — India + relocation preference
  6. Behavioral Signals       (mult)— engagement modifier × base score

Key design choices:
- Title is the decisive dimension: keyword-stuffed skill lists but wrong title = low rank
- Honeypot detection: impossible timelines (exp > tenure) and expert-in-everything profiles
- Behavioral multiplier (not additive): unavailable great-on-paper candidate gets penalized
- Disqualifiers: pure-services background, CV engineering, wrong domain experts
"""

import argparse
import csv
import json
import math
import sys
from datetime import date, datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# JD CONSTANTS  (derived from Senior AI Engineer JD)
# ─────────────────────────────────────────────────────────────────────────────

# Title tier weights — how close is each title to what we need?
TITLE_TIERS = {
    # Tier 1: Exact / near-exact match
    "Senior AI Engineer":               1.00,
    "Lead AI Engineer":                 0.97,
    "Staff Machine Learning Engineer":  0.95,
    "Senior Machine Learning Engineer": 0.93,
    "Senior Applied Scientist":         0.90,
    "Senior NLP Engineer":              0.90,

    # Tier 2: Very strong match
    "AI Engineer":                      0.88,
    "Applied ML Engineer":              0.87,
    "Machine Learning Engineer":        0.86,
    "NLP Engineer":                     0.85,
    "Search Engineer":                  0.84,
    "Recommendation Systems Engineer":  0.84,
    "Senior Data Scientist":            0.80,

    # Tier 3: Strong match with some gap
    "ML Engineer":                      0.78,
    "AI Research Engineer":             0.75,  # possible pure-research flag
    "AI Specialist":                    0.72,
    "Senior Software Engineer (ML)":    0.70,
    "Data Scientist":                   0.68,

    # Tier 4: Adjacent — could work if profile is strong
    "Computer Vision Engineer":         0.40,  # JD explicitly says wrong domain
    "Analytics Engineer":               0.45,
    "Data Engineer":                    0.48,
    "Senior Data Engineer":             0.48,
    "Backend Engineer":                 0.40,
    "Data Analyst":                     0.35,
    "Software Engineer":                0.35,
    "Senior Software Engineer":         0.35,
    "Full Stack Developer":             0.25,
    "Cloud Engineer":                   0.22,
    "DevOps Engineer":                  0.22,
    "Frontend Engineer":                0.18,
    "Java Developer":                   0.15,
    ".NET Developer":                   0.15,
    "Mobile Developer":                 0.15,
    "QA Engineer":                      0.12,
    "Junior ML Engineer":               0.50,  # low but in right domain

    # Tier 5: Wrong domain entirely (JD explicitly calls out)
    "Business Analyst":                 0.05,
    "HR Manager":                       0.04,
    "Mechanical Engineer":              0.03,
    "Accountant":                       0.03,
    "Project Manager":                  0.05,
    "Customer Support":                 0.03,
    "Operations Manager":               0.04,
    "Content Writer":                   0.03,
    "Sales Executive":                  0.03,
    "Civil Engineer":                   0.03,
    "Graphic Designer":                 0.03,
    "Marketing Manager":                0.04,
}

# Core "must-have" skills from JD (embeddings, retrieval, ranking, evaluation)
CORE_SKILLS_REQUIRED = {
    # Retrieval & vector search (highest weight)
    "pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch",
    "elasticsearch", "chroma", "pgvector",
    "vector database", "vector search", "vector db",
    "hybrid search", "dense retrieval", "sparse retrieval",

    # Embedding models
    "sentence-transformers", "sentence transformers", "bge", "e5",
    "text embeddings", "semantic embeddings", "openai embeddings",

    # Ranking & retrieval fundamentals
    "information retrieval", "learning to rank", "neural ranking",
    "bm25", "tfidf", "tf-idf",
    "ndcg", "mrr", "map", "precision@k",
    "ranking evaluation", "retrieval evaluation", "a/b testing",
    "reranking", "re-ranking", "cross-encoder",

    # LLM / AI production
    "llm", "large language model", "fine-tuning", "fine tuning",
    "lora", "qlora", "peft", "rlhf",
    "rag", "retrieval augmented generation",
    "transformers", "huggingface", "hugging face",

    # NLP
    "nlp", "natural language processing", "text classification",
    "named entity recognition", "ner", "question answering",
    "semantic search", "semantic similarity",

    # ML infrastructure
    "mlflow", "weights & biases", "wandb",
    "feature store", "model serving", "model deployment",
    "triton", "torchserve", "bentoml", "vllm",
    "apache beam", "kafka", "spark", "airflow",

    # Python & ML frameworks
    "pytorch", "tensorflow", "jax",
    "xgboost", "lightgbm", "catboost",

    # Recommendation & ranking
    "recommendation system", "recommender system",
    "collaborative filtering", "matrix factorization",
}

# Nice-to-have skills
SKILLS_NICE = {
    "distributed systems", "system design", "microservices",
    "kubernetes", "docker", "aws", "gcp", "azure",
    "python", "sql", "git",
    "open source", "github",
    "data pipeline", "etl",
    "postgresql", "redis",
}

# Explicit red-flag skills (primary domain mismatch)
DOMAIN_MISMATCH_SKILLS = {
    "photoshop", "illustrator", "figma", "sketch", "canva",
    "autocad", "solidworks", "matlab",
    "salesforce", "crm", "erp", "sap",
    "accounting", "tally", "quickbooks",
    "hr management", "payroll",
    "autocad", "civil engineering",
    "unity", "unreal engine",
}

# Companies that signal pure-services background (JD explicitly calls out)
SERVICES_COMPANIES = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "mphasis", "l&t infotech",
    "ltimindtree", "mindtree", "hexaware", "birlasoft", "kpit",
    "niit technologies", "mastech", "syntel", "cyient",
    "persistent systems", "coforge", "zensar", "sonata software",
}

# India cities that are "Tier-1" per JD (Pune, Noida, Hyderabad, Mumbai, Delhi NCR)
PREFERRED_LOCATIONS = {
    "pune", "noida", "hyderabad", "mumbai", "delhi", "bengaluru", "bangalore",
    "chennai", "kolkata", "gurgaon", "gurugram", "greater noida", "ncr", "navi mumbai",
}

# Education tier bonuses
EDU_TIER_BONUS = {
    "tier_1": 0.10,
    "tier_2": 0.05,
    "tier_3": 0.02,
    "tier_4": 0.00,
    "unknown": 0.02,
}

# Reference date for recency calculations
TODAY = date(2026, 6, 27)
TODAY_STR = "2026-06-27"


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def days_since(date_str: str) -> int:
    """Days between date_str (YYYY-MM-DD) and TODAY."""
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return (TODAY - d).days
    except Exception:
        return 9999


def normalize(value: float, lo: float, hi: float) -> float:
    """Clamp-normalize value to [0, 1]."""
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def skill_name_normalized(name: str) -> str:
    return name.lower().strip()


# ─────────────────────────────────────────────────────────────────────────────
# HONEYPOT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def is_honeypot(candidate: dict) -> bool:
    """
    Detect candidates with subtly impossible / implausible profiles.
    JD warns: "8 years at a company founded 3 years ago", "expert in 10 skills with 0 years".
    """
    profile = candidate["profile"]
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    # Check 1: experience duration vs company existence implied by dates
    for job in career:
        start = job.get("start_date", "")
        try:
            start_year = int(start[:4])
            duration_months = job.get("duration_months", 0)
            implied_yoe = duration_months / 12
            # If start year is unreasonably early vs profile total YoE
            if start_year < 1990 and profile["years_of_experience"] < 30:
                return True
        except Exception:
            pass

    # Check 2: Expert in many skills with 0 duration
    expert_zero_duration = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0
    )
    if expert_zero_duration >= 5:
        return True

    # Check 3: years_of_experience impossible given career history dates
    if career:
        try:
            earliest_start = min(
                datetime.strptime(j["start_date"], "%Y-%m-%d").date()
                for j in career if j.get("start_date")
            )
            implied_max_yoe = (TODAY - earliest_start).days / 365.25
            stated_yoe = profile.get("years_of_experience", 0)
            if stated_yoe > implied_max_yoe + 3:  # more than 3 yr gap
                return True
        except Exception:
            pass

    # Check 4: All skills expert with 0 endorsements and 0 duration
    if len(skills) > 0:
        impossible_skills = sum(
            1 for s in skills
            if s.get("proficiency") in ("expert", "advanced")
            and s.get("duration_months", 0) == 0
            and s.get("endorsements", 0) == 0
        )
        if impossible_skills / len(skills) > 0.7 and len(skills) > 5:
            return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT SCORERS
# ─────────────────────────────────────────────────────────────────────────────

def score_title(candidate: dict) -> tuple[float, str]:
    """
    Title/role alignment score (0-1).
    This is the single most important dimension — JD says keyword-stuffed
    candidates in wrong roles must NOT rank high.
    """
    title = candidate["profile"].get("current_title", "")
    base = TITLE_TIERS.get(title, 0.10)

    notes = []

    # Also scan career history for peak relevant role
    career_peak = base
    for job in candidate.get("career_history", []):
        job_title = job.get("title", "")
        t_score = TITLE_TIERS.get(job_title, 0.10)
        if t_score > career_peak:
            career_peak = t_score

    # If current title is lower tier but career shows higher, give partial credit
    if career_peak > base:
        blended = base * 0.6 + career_peak * 0.4
        notes.append(f"career peak: {career_peak:.2f}")
    else:
        blended = base

    # Check summary/headline for role alignment signals
    summary = candidate["profile"].get("summary", "").lower()
    headline = candidate["profile"].get("headline", "").lower()
    combined_text = summary + " " + headline

    # Boost if summary describes AI/ML work even if title is slightly off
    ai_keywords_in_summary = sum(1 for kw in [
        "retrieval", "ranking", "embeddings", "vector", "llm", "nlp",
        "recommendation", "search", "fine-tun", "rag", "semantic"
    ] if kw in combined_text)

    if ai_keywords_in_summary >= 4 and blended < 0.60:
        boost = min(0.15, ai_keywords_in_summary * 0.02)
        blended = min(0.70, blended + boost)
        notes.append(f"summary AI signals: {ai_keywords_in_summary}")

    reason = f"title='{title}' (tier={base:.2f})"
    if notes:
        reason += ", " + ", ".join(notes)

    return blended, reason


def score_skills(candidate: dict) -> tuple[float, str]:
    """
    Skills match score (0-1).
    Weights: proficiency level × duration × endorsements.
    Penalizes pure keyword listing (high proficiency, 0 duration, 0 endorsements).
    """
    skills = candidate.get("skills", [])
    assessment_scores = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})

    if not skills:
        return 0.0, "no skills listed"

    core_matched = []
    nice_matched = []
    mismatch_count = 0

    for skill in skills:
        sname = skill_name_normalized(skill["name"])
        proficiency = skill.get("proficiency", "beginner")
        duration_months = skill.get("duration_months", 0)
        endorsements = skill.get("endorsements", 0)

        # Check if it matches core skills (substring / partial match)
        is_core = any(core in sname or sname in core for core in CORE_SKILLS_REQUIRED)
        is_nice = any(nice in sname or sname in nice for nice in SKILLS_NICE) and not is_core
        is_mismatch = any(mm in sname for mm in DOMAIN_MISMATCH_SKILLS)

        if is_mismatch:
            mismatch_count += 1

        if is_core or is_nice:
            # Proficiency weight
            prof_w = {"beginner": 0.3, "intermediate": 0.6, "advanced": 0.85, "expert": 1.0}.get(proficiency, 0.3)

            # Duration weight: penalize 0-duration claims, reward long usage
            dur_w = 0.0
            if duration_months == 0:
                dur_w = 0.2  # low trust
            elif duration_months <= 6:
                dur_w = 0.5
            elif duration_months <= 24:
                dur_w = 0.75
            else:
                dur_w = 1.0

            # Endorsement signal (0-100 typical range, normalize)
            end_w = min(1.0, endorsements / 30.0)

            # Trust score: blend proficiency, duration, endorsements
            trust = prof_w * 0.5 + dur_w * 0.3 + end_w * 0.2

            # Platform assessment bonus
            assessment_score = assessment_scores.get(skill["name"], -1)
            if assessment_score >= 0:
                assess_w = assessment_score / 100.0
                trust = trust * 0.7 + assess_w * 0.3

            if is_core:
                core_matched.append((skill["name"], trust))
            else:
                nice_matched.append((skill["name"], trust))

    # Score based on number and quality of core matches
    n_core = len(core_matched)
    n_nice = len(nice_matched)

    # Diminishing returns: first 5 core skills most important
    if n_core == 0:
        core_score = 0.0
    else:
        sorted_core = sorted([t for _, t in core_matched], reverse=True)
        # Weight top skills more
        weights = [1.0, 0.9, 0.8, 0.7, 0.6] + [0.4] * max(0, len(sorted_core) - 5)
        weighted_sum = sum(s * w for s, w in zip(sorted_core, weights))
        weight_total = sum(weights[:len(sorted_core)])
        core_score = weighted_sum / weight_total if weight_total > 0 else 0.0

    # Scale by breadth (reward candidates with many relevant skills, up to ~8)
    breadth_bonus = min(1.0, n_core / 6.0)
    core_score = core_score * 0.7 + breadth_bonus * 0.3

    nice_bonus = min(0.10, n_nice * 0.02)

    # Mismatch penalty (lots of wrong-domain skills = weird profile)
    mismatch_penalty = min(0.15, mismatch_count * 0.03)

    raw = core_score + nice_bonus - mismatch_penalty
    final = max(0.0, min(1.0, raw))

    reason = f"{n_core} core AI skills matched"
    if n_core > 0:
        top_names = [n for n, _ in sorted(core_matched, key=lambda x: -x[1])[:4]]
        reason += f" ({', '.join(top_names)})"
    if mismatch_count:
        reason += f"; {mismatch_count} domain-mismatch skills"

    return final, reason


def score_career(candidate: dict) -> tuple[float, str]:
    """
    Career quality score (0-1).
    Rewards: product companies, AI-adjacent industries, healthy tenure.
    Penalizes: all-services career, frequent job-hopping for titles, pure research.
    """
    career = candidate.get("career_history", [])
    if not career:
        return 0.3, "no career history"

    profile = candidate["profile"]
    notes = []
    score = 0.5  # baseline

    # Count time at product vs services companies
    product_months = 0
    services_months = 0
    total_months = 0
    ai_domain_months = 0
    short_tenures = 0

    for job in career:
        company = job.get("company", "").lower()
        duration = job.get("duration_months", 0)
        industry = job.get("industry", "").lower()
        title = job.get("title", "")
        description = job.get("description", "").lower()

        total_months += duration

        # Check if services company
        is_services = any(s in company for s in SERVICES_COMPANIES)
        if is_services:
            services_months += duration
        else:
            product_months += duration

        # AI domain indicators
        ai_domain_keywords = [
            "ai", "machine learning", "ml", "nlp", "search",
            "recommendation", "data science", "analytics", "tech"
        ]
        if any(kw in industry for kw in ai_domain_keywords):
            ai_domain_months += duration

        # Short tenure flag (< 12 months, not current)
        if duration < 12 and not job.get("is_current", False):
            short_tenures += 1

        # Description quality signals
        ai_achievement_keywords = [
            "retrieval", "ranking", "embeddings", "vector", "llm", "fine-tun",
            "rag", "ndcg", "mrr", "a/b test", "production", "shipped", "deployed",
            "recommendation", "search", "recall", "precision"
        ]
        desc_ai_signals = sum(1 for kw in ai_achievement_keywords if kw in description)
        if desc_ai_signals >= 3:
            ai_domain_months += duration // 2  # extra credit for clear AI production work

    # Product vs services ratio
    if total_months > 0:
        product_ratio = product_months / total_months
        services_ratio = services_months / total_months

        if services_ratio > 0.9:
            # Entire career at services — strong disqualifier per JD
            score -= 0.30
            notes.append("all-services career (JD disqualifier)")
        elif services_ratio > 0.6:
            score -= 0.15
            notes.append("majority services background")
        elif product_ratio > 0.7:
            score += 0.15
            notes.append("mostly product companies")

    # Job-hopping flag: more than 3 short tenures
    if short_tenures >= 4:
        score -= 0.10
        notes.append(f"{short_tenures} short tenures (<12mo)")
    elif short_tenures >= 2:
        score -= 0.05

    # Longest tenure (want people who stay)
    max_tenure = max((j.get("duration_months", 0) for j in career), default=0)
    if max_tenure >= 36:
        score += 0.10
        notes.append("demonstrated tenure ≥3yr")
    elif max_tenure < 12:
        score -= 0.10

    # AI industry exposure
    if total_months > 0 and ai_domain_months / total_months > 0.5:
        score += 0.10
        notes.append("majority AI-domain career")

    # Career trajectory: is current role senior to previous?
    current_title = profile.get("current_title", "")
    if current_title:
        current_tier = TITLE_TIERS.get(current_title, 0.10)
        prev_tiers = [TITLE_TIERS.get(j.get("title", ""), 0.10) for j in career if not j.get("is_current")]
        if prev_tiers:
            avg_prev = sum(prev_tiers) / len(prev_tiers)
            if current_tier > avg_prev:
                score += 0.05  # upward trajectory
            elif current_tier < avg_prev - 0.2:
                score -= 0.05  # downward trajectory

    final = max(0.0, min(1.0, score))
    reason = f"product ratio={product_ratio:.0%}" if total_months > 0 else "career analyzed"
    if notes:
        reason += "; " + "; ".join(notes)

    return final, reason


def score_experience(candidate: dict) -> tuple[float, str]:
    """
    Experience calibration score (0-1).
    JD sweet spot: 5-9 years total. Prefers 6-8 years in applied ML.
    Disqualifiers: <2 years AI experience (unless huge total), >15 years "architect" track.
    """
    yoe = candidate["profile"].get("years_of_experience", 0)

    # Gaussian-like curve centered on 7 years, tolerance 3 years
    center = 7.0
    ideal_range_low = 5.0
    ideal_range_high = 9.0

    if ideal_range_low <= yoe <= ideal_range_high:
        # Perfect zone — flat top
        raw = 0.90 + 0.10 * (1 - abs(yoe - center) / (center - ideal_range_low))
    elif yoe < ideal_range_low:
        if yoe < 2:
            raw = 0.10
        else:
            raw = 0.40 + 0.50 * (yoe - 2) / (ideal_range_low - 2)
    else:  # yoe > ideal_range_high
        if yoe > 20:
            raw = 0.40  # over-qualified, probably architect-track
        else:
            raw = 0.90 - 0.30 * (yoe - ideal_range_high) / (20 - ideal_range_high)

    note = f"{yoe:.1f} yrs total"
    if yoe < ideal_range_low:
        note += " (below 5yr floor)"
    elif yoe > ideal_range_high:
        note += " (above 9yr ceiling — may be architect-track)"

    return max(0.0, min(1.0, raw)), note


def score_location(candidate: dict) -> tuple[float, str]:
    """
    Location / logistics score (0-1).
    JD prefers India (Pune/Noida) but accepts Hyderabad, Mumbai, Delhi NCR.
    """
    location = candidate["profile"].get("location", "").lower()
    country = candidate["profile"].get("country", "").lower()
    willing_to_relocate = candidate.get("redrob_signals", {}).get("willing_to_relocate", False)
    notice_period = candidate.get("redrob_signals", {}).get("notice_period_days", 90)
    preferred_mode = candidate.get("redrob_signals", {}).get("preferred_work_mode", "")

    score = 0.0
    notes = []

    # Country check
    if country == "india":
        score += 0.50
        # Location within India
        if any(city in location for city in PREFERRED_LOCATIONS):
            score += 0.40
            notes.append(f"India Tier-1 city ({location})")
        else:
            score += 0.20
            notes.append(f"India ({location})")
    elif willing_to_relocate:
        score += 0.30
        notes.append(f"non-India but willing to relocate ({country})")
    else:
        score += 0.05
        notes.append(f"non-India, not relocating ({country})")

    # Notice period (JD: prefers <30 days, can buy out 30 days)
    if notice_period <= 30:
        score += 0.10
        notes.append("notice ≤30d")
    elif notice_period <= 60:
        score += 0.05
    elif notice_period > 90:
        score -= 0.05
        notes.append(f"long notice ({notice_period}d)")

    # Work mode (JD: hybrid, flexible)
    if preferred_mode in ("hybrid", "flexible", "onsite"):
        pass  # fine
    elif preferred_mode == "remote":
        score -= 0.05  # JD has offices, not fully remote

    final = max(0.0, min(1.0, score))
    return final, "; ".join(notes) if notes else "location analyzed"


def score_behavioral_signals(candidate: dict) -> tuple[float, str]:
    """
    Behavioral signals multiplier (0.3 - 1.2).
    This is a MULTIPLIER applied to base score, not additive.
    Key insight: unavailable/unresponsive perfect-on-paper candidate ≠ hirable.
    """
    sigs = candidate.get("redrob_signals", {})
    notes = []

    multiplier = 1.0

    # ── Availability ──────────────────────────────────────────────────────────
    open_to_work = sigs.get("open_to_work_flag", False)
    last_active_days = days_since(sigs.get("last_active_date", "2020-01-01"))

    if not open_to_work:
        multiplier *= 0.80
        notes.append("not open to work")

    if last_active_days > 180:
        multiplier *= 0.60
        notes.append(f"inactive {last_active_days}d")
    elif last_active_days > 90:
        multiplier *= 0.80
        notes.append(f"inactive {last_active_days}d")
    elif last_active_days <= 30:
        multiplier *= 1.05
        notes.append("active ≤30d")

    # ── Responsiveness ────────────────────────────────────────────────────────
    response_rate = sigs.get("recruiter_response_rate", 0.0)
    avg_resp_time = sigs.get("avg_response_time_hours", 48.0)

    if response_rate >= 0.70:
        multiplier *= 1.05
    elif response_rate < 0.20:
        multiplier *= 0.85
        notes.append(f"low response rate ({response_rate:.0%})")
    elif response_rate < 0.10:
        multiplier *= 0.70
        notes.append(f"very low response rate ({response_rate:.0%})")

    if avg_resp_time < 4:
        multiplier *= 1.02  # very responsive
    elif avg_resp_time > 96:
        multiplier *= 0.95

    # ── Interview reliability ─────────────────────────────────────────────────
    interview_completion = sigs.get("interview_completion_rate", 0.5)
    offer_acceptance = sigs.get("offer_acceptance_rate", -1)

    if interview_completion < 0.50:
        multiplier *= 0.85
        notes.append(f"low interview completion ({interview_completion:.0%})")
    elif interview_completion >= 0.85:
        multiplier *= 1.03

    if offer_acceptance >= 0 and offer_acceptance < 0.30:
        multiplier *= 0.90
        notes.append(f"low offer acceptance ({offer_acceptance:.0%})")

    # ── Platform engagement ───────────────────────────────────────────────────
    profile_completeness = sigs.get("profile_completeness_score", 50.0)
    saved_30d = sigs.get("saved_by_recruiters_30d", 0)

    if profile_completeness >= 85:
        multiplier *= 1.03
    elif profile_completeness < 50:
        multiplier *= 0.92

    if saved_30d >= 5:
        multiplier *= 1.03  # other recruiters are interested too
    
    # ── Verification ─────────────────────────────────────────────────────────
    verified = sum([
        sigs.get("verified_email", False),
        sigs.get("verified_phone", False),
        sigs.get("linkedin_connected", False),
    ])
    if verified == 3:
        multiplier *= 1.02
    elif verified == 0:
        multiplier *= 0.95

    # ── GitHub activity (for AI role, this matters) ───────────────────────────
    github_score = sigs.get("github_activity_score", -1)
    if github_score >= 60:
        multiplier *= 1.05
        notes.append(f"high GitHub activity ({github_score:.0f})")
    elif github_score >= 30:
        multiplier *= 1.02
    elif github_score == -1:
        pass  # no linked GitHub — neutral for non-OSS people

    # Clamp multiplier to reasonable range
    multiplier = max(0.30, min(1.20, multiplier))
    reason = f"engagement mult={multiplier:.2f}" + (f" ({'; '.join(notes)})" if notes else "")
    return multiplier, reason


def score_education(candidate: dict) -> tuple[float, str]:
    """
    Education bonus/penalty (additive modifier, small weight).
    Only a differentiator at the margin; don't over-weight per JD philosophy.
    """
    education = candidate.get("education", [])
    if not education:
        return 0.5, "no education listed"

    best_tier = "unknown"
    has_relevant_degree = False

    relevant_fields = [
        "computer science", "machine learning", "artificial intelligence",
        "data science", "statistics", "mathematics", "information technology",
        "electrical engineering", "electronics"
    ]

    for edu in education:
        tier = edu.get("tier", "unknown")
        field = edu.get("field_of_study", "").lower()
        degree = edu.get("degree", "").lower()

        # Track best tier
        tier_order = ["tier_1", "tier_2", "tier_3", "tier_4", "unknown"]
        if tier_order.index(tier) < tier_order.index(best_tier):
            best_tier = tier

        if any(f in field for f in relevant_fields):
            has_relevant_degree = True

        # Relevant advanced degree
        if any(d in degree for d in ["m.tech", "mtech", "m.e.", "me.", "m.s.", "ms ", "msc", "phd", "m.phil"]):
            if any(f in field for f in relevant_fields):
                has_relevant_degree = True

    score = 0.5 + EDU_TIER_BONUS.get(best_tier, 0.0)
    if has_relevant_degree:
        score += 0.05

    notes = [f"best institution tier={best_tier}"]
    if has_relevant_degree:
        notes.append("relevant degree field")

    return min(1.0, score), "; ".join(notes)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN COMPOSITE SCORER
# ─────────────────────────────────────────────────────────────────────────────

# Component weights — must sum to 1.0 (before behavioral multiplier)
WEIGHTS = {
    "title":      0.35,
    "skills":     0.25,
    "career":     0.20,
    "experience": 0.10,
    "location":   0.05,
    "education":  0.05,
}


def score_candidate(candidate: dict) -> tuple[float, str]:
    """
    Returns (composite_score, reasoning_text).
    composite_score is in [0, 1].
    """
    cid = candidate["candidate_id"]

    # Honeypot check: force to near-zero
    if is_honeypot(candidate):
        return 0.001, "HONEYPOT: impossible profile detected (experience/timeline inconsistency)"

    # Component scores
    t_score, t_note = score_title(candidate)
    s_score, s_note = score_skills(candidate)
    c_score, c_note = score_career(candidate)
    e_score, e_note = score_experience(candidate)
    l_score, l_note = score_location(candidate)
    edu_score, edu_note = score_education(candidate)

    # Weighted base score
    base = (
        WEIGHTS["title"]      * t_score +
        WEIGHTS["skills"]     * s_score +
        WEIGHTS["career"]     * c_score +
        WEIGHTS["experience"] * e_score +
        WEIGHTS["location"]   * l_score +
        WEIGHTS["education"]  * edu_score
    )

    # Behavioral multiplier (from platform signals)
    beh_mult, beh_note = score_behavioral_signals(candidate)
    final = base * beh_mult

    # Clamp
    final = max(0.001, min(0.999, final))

    # Build reasoning (specific, not templated)
    profile = candidate["profile"]
    yoe = profile.get("years_of_experience", 0)
    title = profile.get("current_title", "")
    loc = profile.get("location", "")
    country = profile.get("country", "")

    # Extract key evidence
    top_skills = [s["name"] for s in sorted(
        candidate.get("skills", []),
        key=lambda x: (
            any(core in x["name"].lower() for core in CORE_SKILLS_REQUIRED),
            x.get("duration_months", 0)
        ),
        reverse=True
    )[:4]]

    reasoning = (
        f"{title}, {yoe:.1f}yrs, {loc}/{country}. "
        f"Skills: {', '.join(top_skills[:3])}. "
        f"{t_note}; {s_note}; {beh_note}."
    )
    # Trim long reasonings
    if len(reasoning) > 250:
        reasoning = reasoning[:247] + "..."

    return final, reasoning


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def load_candidates(path: str) -> list[dict]:
    candidates = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


def rank_candidates(candidates: list[dict]) -> list[tuple[str, float, str]]:
    """Returns list of (candidate_id, score, reasoning) sorted by score desc."""
    results = []
    for candidate in candidates:
        score, reasoning = score_candidate(candidate)
        results.append((candidate["candidate_id"], score, reasoning))

    # Sort by score descending; tie-break by candidate_id ascending (per spec)
    # Round scores to 4 decimal places to stabilize floating point ties
    results = [(cid, round(score, 4), reasoning) for cid, score, reasoning in results]
    results.sort(key=lambda x: (-x[1], x[0]))
    return results


def write_submission(ranked: list[tuple[str, float, str]], out_path: str):
    top100 = ranked[:100]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (cid, score, reasoning) in enumerate(top100, start=1):
            writer.writerow([cid, rank, f"{score:.4f}", reasoning])
    print(f"Written {len(top100)} rows to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker")
    parser.add_argument("--candidates", default="candidates.jsonl", help="Path to candidates JSONL")
    parser.add_argument("--out", default="submission.csv", help="Output CSV path")
    parser.add_argument("--debug", action="store_true", help="Print top-20 with details")
    args = parser.parse_args()

    print(f"Loading candidates from {args.candidates}...")
    candidates = load_candidates(args.candidates)
    print(f"Loaded {len(candidates):,} candidates")

    print("Ranking...")
    ranked = rank_candidates(candidates)

    if args.debug:
        print("\n=== TOP 20 ===")
        for i, (cid, score, reasoning) in enumerate(ranked[:20], 1):
            print(f"  {i:2d}. {cid}  score={score:.4f}  {reasoning[:120]}")

        print("\n=== HONEYPOTS IN TOP 200 ===")
        cand_by_id = {c["candidate_id"]: c for c in candidates}
        for cid, score, reasoning in ranked[:200]:
            if "HONEYPOT" in reasoning:
                print(f"  {cid} score={score:.4f} HONEYPOT")

    write_submission(ranked, args.out)
    print("Done.")


if __name__ == "__main__":
    main()
