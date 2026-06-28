# Redrob Intelligent Candidate Ranking System

**Hackathon:** India Runs — Data & AI Challenge (Redrob)  
**Challenge:** Intelligent Candidate Discovery & Ranking  
**Task:** Rank top 100 candidates from 100K pool for a Senior AI Engineer (Founding Team) role

---

## TL;DR

```bash
pip install -r requirements.txt
python rank.py --candidates candidates.jsonl --out submission.csv
python validate_submission.py submission.csv
```

Runs in ~50 seconds on CPU. No GPU, no API calls, no network required.

---

## Approach

### Philosophy

> "The right answer is not 'find candidates whose skills section contains the most AI keywords.' That's a trap." — Challenge JD

The system ranks candidates the way a senior recruiter who has read the JD *carefully* would — not by counting keywords, but by understanding what the role actually needs.

### Architecture: Multi-Signal Hybrid Scoring

Five component scores are computed for each candidate, combined with a behavioral multiplier:

```
Final Score = Base Score × Behavioral Multiplier

Base Score = weighted sum of:
  - Title / Role Alignment   (35%)
  - Skills Match             (25%)
  - Career Quality           (20%)
  - Experience Calibration   (10%)
  - Location & Logistics      (5%)
  - Education Signal          (5%)
```

### Key Design Decisions

**1. Title is the decisive dimension (35% weight)**  
The JD explicitly says keyword-stuffed HR Managers or Marketing Managers with AI skills on their profile are not fits. An explicit title tier map separates roles by how close they are to "Senior AI Engineer" — from tier-1 (exact match) to tier-5 (0.03-0.05 for wrong-domain roles). A candidate with the perfect skill list but the title "Business Analyst" gets a near-zero title score, which suppresses their overall score well below any true AI Engineer.

**2. Skills trust scoring, not keyword matching (25% weight)**  
For each skill, we compute a *trust score* that blends:
- Proficiency level (beginner → expert)
- Duration of use (0 months = low trust, 2+ years = full trust)
- Peer endorsements on the platform
- Redrob assessment scores (when available)

This catches "keyword stuffers" — candidates who list every AI framework at "expert" with 0 months duration and 0 endorsements get low trust scores on those skills.

**3. Career quality penalizes services firms and job-hoppers (20% weight)**  
The JD explicitly disqualifies people whose entire career is at TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, etc. We compute the ratio of career time at product companies vs services companies and penalize heavily for >90% services backgrounds. Job-hopping (4+ tenures under 12 months) is also penalized.

**4. Behavioral signals are a multiplier, not additive**  
A perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% recruiter response rate is, for hiring purposes, **not actually available**. Behavioral signals modify the base score multiplicatively (range 0.3–1.2x):
- `open_to_work_flag = False` → 0.80x
- `last_active_date` > 180 days ago → 0.60x
- `recruiter_response_rate` < 20% → 0.85x
- `interview_completion_rate` < 50% → 0.85x
- GitHub activity score ≥ 60 → 1.05x
- All three verifications + active in last 30 days → 1.10x cumulative

**5. Honeypot detection**  
The dataset contains ~80 honeypots with impossible profiles. We detect:
- Expert-level proficiency on many skills with 0 duration AND 0 endorsements
- Stated years of experience that exceed what career history dates allow
- Career timeline impossibilities (company founded later than stated start)

Detected honeypots receive score ≈ 0.001 and appear well below rank 100.

---

## Scoring Details

### Title Tier Map

| Tier | Titles | Score |
|------|--------|-------|
| 1 | Senior AI Engineer, Lead AI Engineer, Staff MLE | 0.93–1.00 |
| 2 | AI Engineer, Applied ML Engineer, Machine Learning Engineer, NLP Engineer, Search Engineer, Recommendation Systems Engineer | 0.84–0.90 |
| 3 | ML Engineer, AI Research Engineer, AI Specialist, Data Scientist | 0.68–0.78 |
| 4 | Analytics Engineer, Data Engineer, Backend Engineer, Software Engineer | 0.22–0.50 |
| 5 | Business Analyst, HR Manager, Mechanical Engineer, Accountant, etc. | 0.03–0.05 |

### Core Skills (required per JD)

Vector DBs: Pinecone, Weaviate, Qdrant, Milvus, FAISS, OpenSearch, Elasticsearch, pgvector  
Embeddings: Sentence Transformers, BGE, E5, Hugging Face  
Retrieval: BM25, hybrid search, dense retrieval, information retrieval  
Evaluation: NDCG, MRR, MAP, A/B testing, reranking  
LLM/Finetune: LoRA, QLoRA, PEFT, RAG, fine-tuning  
NLP: NLP, semantic search, semantic similarity, text classification  
ML Frameworks: PyTorch, TensorFlow, XGBoost  

---

## Runtime Characteristics

| Metric | Value |
|--------|-------|
| Candidates | 100,000 |
| Runtime (CPU-only) | ~50 seconds |
| Memory | < 1 GB |
| Network required | No |
| GPU required | No |
| API calls | None |

All computation is local, rule-based + deterministic. No model weights to download.

---

## Repository Structure

```
.
├── rank.py                    # Main ranking script
├── requirements.txt           # Dependencies (minimal)
├── validate_submission.py     # Official submission validator
├── submission_metadata.yaml   # Team metadata
├── candidates.jsonl           # Input data (not committed; download separately)
└── README.md                  # This file
```

---

## Reproducing the Submission

**Step 1:** Download `candidates.jsonl` from the challenge data bundle and place it in this directory.

**Step 2:** Install dependencies:
```bash
pip install -r requirements.txt
```

**Step 3:** Run the ranker:
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

**Step 4:** Validate:
```bash
python validate_submission.py submission.csv
```

Expected output: `Submission is valid.`

---

## Design Trade-offs

**Why not use semantic embeddings?**  
The compute constraint (5 min CPU, 16 GB RAM, no network) makes embedding 100K candidates infeasible at Stage 3 reproduction without pre-computed indexes. The rule-based approach achieves similar recall for the right-domain candidates because the JD's discriminating signals (title, core skill names, career type) are structured enough to capture analytically — they don't require semantic similarity.

**Why is title weighted 35%?**  
The JD explicitly warns against keyword stuffers. Testing showed that a candidate with the title "Marketing Manager" who lists every AI framework in their skills still reads as a wrong-domain fit to the recruiter. Title is the strongest single signal and deserves dominant weight.

**Why is behavioral signal a multiplier, not additive?**  
A great-on-paper candidate who is unreachable is worth *less than* a slightly-less-qualified candidate who is active, responsive, and engaged. Making behavioral signals additive would allow an unavailable star to still rank above an available strong fit. Multiplicative behavior captures the "unreachable → not actually hirable" insight.
