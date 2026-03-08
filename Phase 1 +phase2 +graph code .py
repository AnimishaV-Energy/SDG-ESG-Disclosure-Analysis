#!/usr/bin/env python
# coding: utf-8

# In[28]:


# ===========================
# PHASE 1 ONLY (AUTO-VERSIONED OUTPUT to Downloads + Excel-safe)
# ===========================

import os, re, time, logging
from collections import defaultdict, Counter
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import fitz  # PyMuPDF

# --- Optional: SentenceTransformer ---
ST_AVAILABLE = True
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    ST_AVAILABLE = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ===========================
# USER PATHS
# ===========================
#PDF_PATH = "/Users/abc/Desktop/IPS/Annual Report _2023-24.pdf"
#SDG_INPUT_XLSX = "/Users/abc/Desktop/IPS/SDG_description_Model_data.xlsx"
PDF_PATH = r"/Users/abc/Downloads/Annual Report 2023-24.pdf"
SDG_INPUT_XLSX = r"/Users/abc/Desktop/IPS/SDG_description_Model_data.xlsx"


# ✅ Save output to Downloads
#OUT_DIR = "/Users/abc/Desktop/IPS"
OUT_DIR = r"/Users/abc/Downloads/IPS/SDG_Output"
os.makedirs(OUT_DIR, exist_ok=True)


# ===========================
# SETTINGS
# ===========================
SDG_FILTER_PREFIX = ("SDG 6", "SDG 7", "SDG 8", "SDG 11", "SDG 13")

SIMILARITY_THRESHOLD = 0.23
WINDOW_SIZE = 3
TOP_N_PER_SDG = 20
TOP_N_KEYWORDS_OUT = 25
MIN_BLOCK_LEN = 200
MIN_SENT_LEN = 20
SKIP_FINANCIALS = True
USE_TARGET_LEVEL = True


# ===========================
# LOGGING
# ===========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ===========================
# AUTO-INCREMENT FILE NAMING
# ===========================
def get_next_versioned_filename(out_dir: str, base_name: str, ext: str = ".xlsx", pad: int = 3) -> str:
    """
    Generates:
    SDG_Dictionary_001.xlsx, SDG_Dictionary_002.xlsx, ...
    """
    pattern = re.compile(rf"^{re.escape(base_name)}_(\d+){re.escape(ext)}$", re.IGNORECASE)
    nums = []
    for f in os.listdir(out_dir):
        m = pattern.match(f)
        if m:
            nums.append(int(m.group(1)))
    next_num = (max(nums) + 1) if nums else 1
    return os.path.join(out_dir, f"{base_name}_{next_num:0{pad}d}{ext}")


PHASE1_OUT_XLSX = get_next_versioned_filename(OUT_DIR, base_name="SDG_Dictionary", ext=".xlsx", pad=3)
logging.info("📁 Phase-1 output will be saved to: %s", PHASE1_OUT_XLSX)


# ===========================
# EXCEL-SAFE CLEANING (Fix IllegalCharacterError)
# ===========================
ILLEGAL_EXCEL_RE = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]")  # illegal control chars

def excel_safe(x, max_len=32000) -> str:
    """Remove illegal control chars + keep under Excel cell limit."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x)
    s = ILLEGAL_EXCEL_RE.sub("", s)
    if len(s) > max_len:
        s = s[:max_len]
    return s

def clean_df_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    """Clean only object columns to avoid IllegalCharacterError."""
    df2 = df.copy()
    obj_cols = df2.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        df2[c] = df2[c].map(excel_safe)
    return df2


# ===========================
# TEXT HELPERS
# ===========================
def norm_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\u00ad", "")         # soft hyphen
    s = re.sub(r"-\s*\n", "", s)        # hyphen line breaks
    s = re.sub(r"\s+", " ", s)
    s = s.strip()
    # remove illegal excel chars early (extra safe)
    s = ILLEGAL_EXCEL_RE.sub("", s)
    return s

def split_into_sentences(text: str) -> List[str]:
    text = norm_text(text)
    sents = re.split(r"(?<=[.!?])\s+", text)
    sents = [x.strip() for x in sents if len(x.strip()) >= MIN_SENT_LEN]
    return sents if sents else ([text] if text else [])

def sliding_windows(sentences: List[str], w: int) -> List[Dict[str, Any]]:
    if not sentences:
        return []
    w = max(1, min(w, len(sentences)))
    return [{"text": " ".join(sentences[i:i+w]), "range": (i, i+w-1), "count": w}
            for i in range(len(sentences)-w+1)]

def parse_keywords(cell) -> List[str]:
    if pd.isna(cell):
        return []
    parts = re.split(r"[,\n;|]+", str(cell))
    return [p.strip() for p in parts if p.strip()]

def extract_last_page_number(line: str) -> Optional[int]:
    nums = re.findall(r"\b(\d{1,4})\b", line)
    return int(nums[-1]) if nums else None

def find_pages_for_patterns(toc_text: str, patterns: List[str]) -> List[int]:
    pages = []
    for raw in toc_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        for pat in patterns:
            if re.search(pat, line, flags=re.IGNORECASE):
                pno = extract_last_page_number(line)
                if pno is not None:
                    pages.append(pno)
    return pages

def find_nearby_pdf_page(doc: fitz.Document, marker_patterns: List[str], around_pdf_page: int, radius: int = 300) -> Optional[int]:
    lo = max(1, around_pdf_page - radius)
    hi = min(doc.page_count, around_pdf_page + radius)
    for p in range(lo, hi + 1):
        txt = norm_text(doc.load_page(p-1).get_text("text")).lower()
        if any(re.search(pat, txt, flags=re.IGNORECASE) for pat in marker_patterns):
            return p
    return None


# ===========================
# LOAD SDG CONFIG
# ===========================
def load_sdg_mapping(xlsx_path: str):
    df = pd.read_excel(xlsx_path, sheet_name=0, engine="openpyxl")
    df.columns = df.columns.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    rename_map = {
        "target _Indicator_description": "target_Indicator_description",
        "target Indicator description": "target_Indicator_description",
        "track (proxie(P) / Report(R) )": "track (proxie(P) / Report(R))",
    }
    df = df.rename(columns=lambda c: rename_map.get(c, c))

    required_cols = {
        "SDG", "SDG_Description", "type", "target/Indicator",
        "target_Indicator_description", "NTPC Keywords", "track (proxie(P) / Report(R))"
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in SDG mapping: {missing}. Found: {list(df.columns)}")

    df["SDG"] = df["SDG"].astype(str).str.strip()
    df = df[df["SDG"].str.startswith(SDG_FILTER_PREFIX, na=False)].copy()

    logging.info("✅ SDGs loaded: %s", sorted(df["SDG"].unique().tolist()))
    logging.info("✅ Rows after filter: %d", len(df))

    sdg_keywords = defaultdict(set)
    sdg_targets = defaultdict(list)

    for _, r in df.iterrows():
        sdg = str(r["SDG"]).strip()
        ctx = norm_text(f"{r.get('SDG_Description','')}. {r.get('target_Indicator_description','')}")
        kws = parse_keywords(r.get("NTPC Keywords", ""))

        target = str(r.get("target/Indicator", "")).strip()
        target_desc = str(r.get("target_Indicator_description", "")).strip()
        typ = str(r.get("type", "")).strip()
        track = str(r.get("track (proxie(P) / Report(R))", "")).strip()

        for k in kws:
            sdg_keywords[sdg].add(k)

        sdg_targets[sdg].append((target, target_desc, typ, track, ctx))

    sdg_keywords = {k: sorted(list(v)) for k, v in sdg_keywords.items()}
    return sdg_keywords, sdg_targets


# ===========================
# FINANCIAL SKIP
# ===========================
TOC_MARKERS = ("contents", "content index", "table of contents")

SFS_PATTERNS = [r"\bstandalone\s+financial\s+statements\b", r"\bfinancial\s+statements\s*\(sfs\)\b"]
CFS_PATTERNS = [r"\bconsolidated\s+financial\s+statements\b", r"\bfinancial\s+statements\s*\(cfs\)\b"]

AFTER_FIN_PATTERNS = [
    r"\bindependent\s+assurance\s+report\b",
    r"\bgri\s+index\b",
    r"\bun\s+sdg\s+index\b",
    r"\bbusiness\s+responsibil\w+\s+and\s+sustainabil\w+\s+report\b",
    r"\bdirectors['’]\s+report\b",
    r"\bnotice\b.*\bannual\s+general\s+meeting\b",
    r"\bnotice\s+of\b.*\bagm\b",
    r"\breport\s+on\s+corporate\s+governance\b",
    r"\bmanagement\s+discussion\b",
]

def extract_toc_text(doc: fitz.Document, scan_pages: int = 120) -> str:
    toc_pages = []
    limit = min(scan_pages, doc.page_count)
    for p in range(1, limit + 1):
        txt = doc.load_page(p-1).get_text("text").lower()
        if any(m in txt for m in TOC_MARKERS):
            toc_pages.append(p)

    if toc_pages:
        start = min(toc_pages)
        end = min(max(toc_pages) + 8, doc.page_count)
    else:
        start, end = 1, min(scan_pages, doc.page_count)

    buf = []
    for p in range(start, end + 1):
        buf.append(norm_text(doc.load_page(p-1).get_text("text")))
    return "\n".join(buf)

def get_skip_range_financials(pdf_path: str):
    doc = fitz.open(pdf_path)
    toc_text = extract_toc_text(doc, scan_pages=120)

    fin_candidates = find_pages_for_patterns(toc_text, SFS_PATTERNS) + find_pages_for_patterns(toc_text, CFS_PATTERNS)
    if not fin_candidates:
        raise ValueError("TOC parse failed: Financial Statements not found.")

    fin_start_printed = min(fin_candidates)
    fin_anchor_printed = max(fin_candidates)

    after_pages = find_pages_for_patterns(toc_text, AFTER_FIN_PATTERNS)
    after_pages = [p for p in after_pages if p and p > fin_anchor_printed]
    fin_end_printed = (min(after_pages) - 1) if after_pages else None

    pdf_fin_start = find_nearby_pdf_page(
        doc,
        marker_patterns=[r"standalone financial statements", r"consolidated financial statements", r"financial statements"],
        around_pdf_page=fin_start_printed
    )

    pdf_next_start = None
    if fin_end_printed is not None:
        pdf_next_start = find_nearby_pdf_page(doc, AFTER_FIN_PATTERNS, fin_end_printed + 1)

    offsets = []
    if pdf_fin_start:
        offsets.append(pdf_fin_start - fin_start_printed)
    if pdf_next_start and fin_end_printed is not None:
        offsets.append(pdf_next_start - (fin_end_printed + 1))

    offset_used = int(round(sum(offsets) / len(offsets))) if offsets else 0

    skip_start = fin_start_printed + offset_used
    skip_end = (fin_end_printed + offset_used) if fin_end_printed is not None else doc.page_count

    skip_start = max(1, min(skip_start, doc.page_count))
    skip_end = max(1, min(skip_end, doc.page_count))
    if skip_end < skip_start:
        return (0, -1)

    return (skip_start, skip_end)


# ===========================
# EXTRACT BLOCKS
# ===========================
def extract_blocks(pdf_path: str, skip_range: Optional[Tuple[int, int]]):
    doc = fitz.open(pdf_path)
    recs = []
    skip_start, skip_end = skip_range if skip_range else (0, -1)

    for page_no in range(1, doc.page_count + 1):
        if skip_start <= page_no <= skip_end:
            continue
        page = doc.load_page(page_no - 1)
        for b_idx, b in enumerate(page.get_text("blocks")):
            txt = norm_text(b[4])
            if len(txt) >= MIN_BLOCK_LEN:
                recs.append({"page": page_no, "block_id": f"{page_no}_{b_idx}", "text": txt})
    return recs


# ===========================
# SIMILARITY ENGINES
# ===========================
def try_load_sentence_transformer(model_name: str):
    if not ST_AVAILABLE:
        return None
    try:
        return SentenceTransformer(model_name)
    except Exception as e:
        logging.warning("⚠️ SentenceTransformer failed: %s", e)
        return None

def tfidf_similarity_score(texts: List[str], query: str) -> np.ndarray:
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform(texts + [query])
    sims = cosine_similarity(X[:-1], X[-1])
    return sims.reshape(-1)


# ===========================
# PHASE 1: BUILD OUTPUT
# ===========================
def phase1_build_dictionary():
    t0 = time.time()

    logging.info("📌 Phase-1: Loading SDG mapping...")
    sdg_keywords, sdg_targets = load_sdg_mapping(SDG_INPUT_XLSX)

    # Financial skip
    skip_range = None
    if SKIP_FINANCIALS:
        try:
            skip_range = get_skip_range_financials(PDF_PATH)
            logging.info("✅ Skipping Financial Statements pages (PDF): %s to %s", skip_range[0], skip_range[1])
        except Exception as ex:
            logging.warning("⚠️ Financial skip failed; continuing without skipping: %s", ex)

    # Extract PDF blocks
    logging.info("📌 Extracting blocks from PDF...")
    blocks = extract_blocks(PDF_PATH, skip_range)
    logging.info("✅ Blocks extracted: %d", len(blocks))

    # Keyword gate
    logging.info("📌 Keyword-gating blocks...")
    gated_blocks = []
    for blk in blocks:
        t = blk["text"].lower()
        sdg_hits = {}
        for sdg, kws in sdg_keywords.items():
            hits = [k for k in kws if k.lower() in t]
            if hits:
                sdg_hits[sdg] = hits
        if sdg_hits:
            blk["sdg_hits"] = sdg_hits
            gated_blocks.append(blk)

    logging.info("✅ Blocks after keyword gate: %d", len(gated_blocks))
    logging.info("⏳ Starting similarity scan over %d blocks...", len(gated_blocks))

    # Build SDG context strings
    sdg_context = {}
    for sdg, targets in sdg_targets.items():
        joined = norm_text(" ".join([t[4] for t in targets if t[4]]))
        sdg_context[sdg] = joined if joined else sdg

    # Load embedder if available
    embedder = try_load_sentence_transformer("all-MiniLM-L6-v2")
    use_st = embedder is not None
    logging.info("✅ Similarity engine: %s", "SentenceTransformer" if use_st else "TF-IDF (fallback)")

    # Precompute SDG query embeddings once
    sdg_query_emb = {}
    if use_st:
        for sdg, q in sdg_context.items():
            sdg_query_emb[sdg] = embedder.encode([q], normalize_embeddings=True, show_progress_bar=False)[0]

    # Precompute target embeddings once per SDG (for target-level selection)
    sdg_target_ctxs = {}
    sdg_target_meta = {}
    sdg_target_emb = {}
    if use_st and USE_TARGET_LEVEL:
        for sdg, targets in sdg_targets.items():
            target_ctxs = [t[4] for t in targets]
            sdg_target_ctxs[sdg] = target_ctxs
            sdg_target_meta[sdg] = targets
            if target_ctxs:
                sdg_target_emb[sdg] = embedder.encode(target_ctxs, normalize_embeddings=True, show_progress_bar=False)

    rows = []

    # Main scan loop
    for i, blk in enumerate(gated_blocks, start=1):
        sents = split_into_sentences(blk["text"])
        wins = sliding_windows(sents, WINDOW_SIZE)
        if not wins:
            continue

        win_texts = [w["text"] for w in wins]

        # Encode windows once per block (ST only)
        win_emb = None
        if use_st:
            win_emb = embedder.encode(win_texts, normalize_embeddings=True, show_progress_bar=False)

        for sdg, hits in blk["sdg_hits"].items():
            # Similarity per window
            if use_st:
                sims = np.dot(win_emb, sdg_query_emb[sdg])
            else:
                sims = tfidf_similarity_score(win_texts, sdg_context[sdg])

            for idx, (w, sim) in enumerate(zip(wins, sims)):
                if float(sim) < SIMILARITY_THRESHOLD:
                    continue

                best_target = best_desc = best_type = best_track = ""
                best_tsim = None

                # Target-level selection
                if USE_TARGET_LEVEL and sdg_targets.get(sdg):
                    if use_st and sdg in sdg_target_emb and sdg_target_emb[sdg].shape[0] > 0:
                        w_vec = win_emb[idx]
                        tsims = np.dot(sdg_target_emb[sdg], w_vec)
                        j = int(np.argmax(tsims))
                        best_tsim = float(tsims[j])
                        best_target, best_desc, best_type, best_track, _ = sdg_target_meta[sdg][j]
                    else:
                        target_ctxs = [t[4] for t in sdg_targets[sdg]]
                        tsims = tfidf_similarity_score(target_ctxs, w["text"])
                        j = int(np.argmax(tsims))
                        best_tsim = float(tsims[j])
                        best_target, best_desc, best_type, best_track, _ = sdg_targets[sdg][j]
                else:
                    if sdg_targets.get(sdg):
                        best_target, best_desc, best_type, best_track, _ = sdg_targets[sdg][0]

                rows.append({
                    "SDG": sdg,
                    "Page": blk["page"],
                    "Sentence": w["text"],
                    "Sentence_Count": w["count"],
                    "SDG_Target_Indicator": best_target,
                    "SDG_Target_Indicator_Description": best_desc,
                    "Type": best_type,
                    "Track": best_track,
                    "Keywords_Fetched": ", ".join(hits),
                    "Semantic_Similarity": round(float(sim), 3),
                    "Target_Similarity": round(best_tsim, 3) if best_tsim is not None else ""
                })

        if i % 50 == 0:
            logging.info("…processed %d/%d blocks | elapsed %.1f min", i, len(gated_blocks), (time.time() - t0) / 60)

    # Detail output
    df_detail = pd.DataFrame(rows).drop_duplicates(subset=["SDG", "Page", "Sentence"])

    if df_detail.empty:
        logging.warning("⚠️ No SDG matches found. Try lowering SIMILARITY_THRESHOLD or MIN_BLOCK_LEN.")
        df_dictionary = pd.DataFrame(columns=["SDG", "Context", "Keywords"])
    else:
        # Impact score
        df_detail["Keyword_Hit_Count"] = df_detail["Keywords_Fetched"].fillna("").apply(
            lambda x: len([k for k in str(x).split(",") if k.strip()])
        )
        df_detail["Impact_Score"] = df_detail["Semantic_Similarity"].astype(float) + 0.02 * df_detail["Keyword_Hit_Count"].astype(float)

        # Top N per SDG
        df_top = (
            df_detail.sort_values(["SDG", "Impact_Score"], ascending=[True, False])
                     .groupby("SDG", as_index=False).head(TOP_N_PER_SDG)
                     .copy()
        )

        # Dictionary sheet
        dict_rows = []
        for sdg, grp in df_top.groupby("SDG"):
            grp = grp.sort_values("Impact_Score", ascending=False)
            context = "\n\n".join([f"{i+1}. {s}" for i, s in enumerate(grp["Sentence"].tolist())])

            hits_all = []
            for cell in grp["Keywords_Fetched"].fillna("").tolist():
                hits_all.extend([k.strip().lower() for k in str(cell).split(",") if k.strip()])
            freq = Counter(hits_all)
            keywords_out = ", ".join([k for k, _ in freq.most_common(TOP_N_KEYWORDS_OUT)])
            dict_rows.append({"SDG": sdg, "Context": context, "Keywords": keywords_out})

        df_dictionary = pd.DataFrame(dict_rows)
        df_detail = df_top.drop(columns=["Keyword_Hit_Count"])

    # ✅ CLEAN before writing (fix IllegalCharacterError)
    df_dictionary_x = clean_df_for_excel(df_dictionary)
    df_detail_x = clean_df_for_excel(df_detail)

    logging.info("📌 Writing Phase-1 output: %s", PHASE1_OUT_XLSX)
    with pd.ExcelWriter(PHASE1_OUT_XLSX, engine="openpyxl") as writer:
        df_dictionary_x.to_excel(writer, sheet_name="Dictionary", index=False)
        df_detail_x.to_excel(writer, sheet_name="SDG", index=False)

    logging.info("✅ Phase-1 done.")
    logging.info("✅ Dictionary rows: %d", len(df_dictionary_x))
    logging.info("✅ Detail rows: %d", len(df_detail_x))

    return PHASE1_OUT_XLSX


# ===========================
# RUN PHASE 1 ONLY
# ===========================
print("✅ Starting Phase-1 ...")
phase1_file = phase1_build_dictionary()
print("🎉 COMPLETED")
print("Phase-1 output:", phase1_file)


# In[34]:


get_ipython().system('curl http://localhost:11434/api/tags')


# In[1]:


get_ipython().system('pip install ollama')


# In[2]:


import os, json, re, time
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np

from ollama import chat  # official client [5](https://github.com/ollama/ollama-python)

# =========================
# PATHS (edit only if needed)
# =========================
PHASE1_XLSX = Path("/Users/abc/Downloads/IPS/SDG_Output/SDG_Dictionary_001.xlsx")
SDG_MAPPING_XLSX = Path("/Users/abc/Downloads/IPS/SDG_description_Model_data.xlsx")  # optional but recommended
OUT_DIR = Path("/Users/abc/Downloads/IPS/SDG_Output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PHASE2_XLSX = OUT_DIR / "SDG_Phase2_GapAnalysis_001.xlsx"

# =========================
# OLLAMA MODEL
# =========================
MODEL = "qwen2:latest"   # already installed from your ollama list
TEMPERATURE = 0.2

# =========================
# LOAD PHASE-1
# =========================
df_dict = pd.read_excel(PHASE1_XLSX, sheet_name="Dictionary", engine="openpyxl")
df_sdg  = pd.read_excel(PHASE1_XLSX, sheet_name="SDG", engine="openpyxl")

# optional mapping for missing targets
df_map = None
if SDG_MAPPING_XLSX.exists():
    df_map = pd.read_excel(SDG_MAPPING_XLSX, sheet_name=0, engine="openpyxl")
    df_map.columns = df_map.columns.astype(str).str.strip()

# =========================
# DETERMINISTIC COVERAGE SCORING
# =========================
def strength_bucket(n_evidence, avg_sim):
    if n_evidence >= 8 and avg_sim >= 0.30: return "Strong"
    if n_evidence >= 3 and avg_sim >= 0.25: return "Moderate"
    if n_evidence >= 1: return "Weak"
    return "None"

cov = []
for (sdg, target), g in df_sdg.groupby(["SDG", "SDG_Target_Indicator"], dropna=False):
    n = len(g)
    avg_sim = pd.to_numeric(g["Semantic_Similarity"], errors="coerce").fillna(0).mean()
    pages = sorted(set(g["Page"].tolist()))
    cov.append({
        "SDG": sdg,
        "Target_Indicator": target,
        "Evidence_Count": n,
        "Avg_Semantic_Similarity": round(float(avg_sim), 3),
        "Pages": ", ".join(map(str, pages[:40])) + ("..." if len(pages) > 40 else ""),
        "Strength": strength_bucket(n, avg_sim)
    })

df_coverage = pd.DataFrame(cov).sort_values(["SDG","Strength","Evidence_Count"], ascending=[True, True, False])

# Missing targets (if mapping exists)
df_missing = pd.DataFrame()
if df_map is not None and "SDG" in df_map.columns and "target/Indicator" in df_map.columns:
    mapped = df_map.copy()
    mapped["SDG"] = mapped["SDG"].astype(str).str.strip()
    mapped["target/Indicator"] = mapped["target/Indicator"].astype(str).str.strip()

    # focus on SDG 6/7/8/11/13
    mapped = mapped[mapped["SDG"].str.startswith(("SDG 6","SDG 7","SDG 8","SDG 11","SDG 13"), na=False)]

    present = set(zip(df_coverage["SDG"].astype(str), df_coverage["Target_Indicator"].astype(str)))
    miss = []
    for _, r in mapped.iterrows():
        k = (r["SDG"], r["target/Indicator"])
        if k not in present:
            miss.append({
                "SDG": r["SDG"],
                "Target_Indicator": r["target/Indicator"],
                "Target_Description": str(r.get("target_Indicator_description","")).strip(),
                "Gap_Type": "No evidence found in report (Phase-1)"
            })
    df_missing = pd.DataFrame(miss)

# =========================
# OLLAMA DEEP ANALYSIS PROMPT
# =========================
SYSTEM = (
    "You are an ESG/SDG reporting analyst. "
    "Use ONLY the evidence snippets provided (page + text). "
    "If evidence is insufficient, say so clearly. "
    "Return STRICT JSON only (no markdown)."
)

def safe_json_extract(text):
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except:
        return None

def analyze_one_sdg(sdg, context_text, ev_df):
    top = ev_df.sort_values("Semantic_Similarity", ascending=False).head(20)
    evidence = [{"page": int(r["Page"]), "text": str(r["Sentence"])} for _, r in top.iterrows()]

    payload = {
        "sdg": sdg,
        "context_from_phase1_dictionary": str(context_text)[:6000],
        "evidence": evidence
    }

    prompt = f"""
Return STRICT JSON with keys:
summary,
key_claims (list of strings),
evidence_quality (Strong/Moderate/Weak),
gaps (list of objects with fields: gap, why_it_matters, what_to_add),
kpis_to_add (list),
rewrite_suggestions (list),
risk_and_governance_gaps (list).

Input:
{json.dumps(payload, ensure_ascii=False)}
"""

    resp = chat(
        model=MODEL,
        messages=[
            {"role":"system","content": SYSTEM},
            {"role":"user","content": prompt}
        ],
        options={"temperature": TEMPERATURE},
    )
    out = resp["message"]["content"]
    obj = safe_json_extract(out)
    if obj is None:
        obj = {"sdg": sdg, "parse_error": True, "raw": out}
    obj["sdg"] = sdg
    return obj

# =========================
# RUN SDG-LEVEL ANALYSIS
# =========================
sdg_reports = []
for _, r in df_dict.iterrows():
    sdg = r["SDG"]
    ctx = r.get("Context","")
    ev = df_sdg[df_sdg["SDG"] == sdg].copy()
    if ev.empty:
        sdg_reports.append({"sdg": sdg, "summary": "No evidence in Phase-1 output.", "gaps": [], "kpis_to_add": []})
    else:
        sdg_reports.append(analyze_one_sdg(sdg, ctx, ev))

df_exec = pd.json_normalize(sdg_reports)

gap_rows, rec_rows = [], []
for rep in sdg_reports:
    sdg = rep.get("sdg")
    for g in rep.get("gaps", []) or []:
        gap_rows.append({"SDG": sdg, **g})
    for k in rep.get("kpis_to_add", []) or []:
        rec_rows.append({"SDG": sdg, "Type": "KPI", "Recommendation": k})
    for s in rep.get("rewrite_suggestions", []) or []:
        rec_rows.append({"SDG": sdg, "Type": "Rewrite", "Recommendation": s})
    for rg in rep.get("risk_and_governance_gaps", []) or []:
        rec_rows.append({"SDG": sdg, "Type": "Governance/Risk", "Recommendation": rg})

df_gaps = pd.DataFrame(gap_rows)
df_recs = pd.DataFrame(rec_rows)

# =========================
# WRITE OUTPUT
# =========================
with pd.ExcelWriter(OUT_PHASE2_XLSX, engine="openpyxl") as w:
    df_exec.to_excel(w, sheet_name="SDG_Executive_Summary", index=False)
    df_coverage.to_excel(w, sheet_name="Target_Coverage", index=False)
    if not df_missing.empty:
        df_missing.to_excel(w, sheet_name="Missing_Targets", index=False)
    df_sdg.to_excel(w, sheet_name="Evidence_All", index=False)
    df_gaps.to_excel(w, sheet_name="Gaps", index=False)
    df_recs.to_excel(w, sheet_name="Recommendations", index=False)

print("✅ Phase-2 complete:", OUT_PHASE2_XLSX)


# In[3]:


import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# FILE PATHS (edit if needed)
# -------------------------
PHASE2_XLSX = Path("/Users/abc/Downloads/IPS/SDG_Output/SDG_Phase2_GapAnalysis_001.xlsx")

OUT_DIR = Path("/Users/abc/Downloads/IPS/SDG_Output/Phase2_Charts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# LOAD SHEETS
# -------------------------
df_exec = pd.read_excel(PHASE2_XLSX, sheet_name="SDG_Executive_Summary", engine="openpyxl")
df_cov  = pd.read_excel(PHASE2_XLSX, sheet_name="Target_Coverage", engine="openpyxl")
df_ev   = pd.read_excel(PHASE2_XLSX, sheet_name="Evidence_All", engine="openpyxl")
df_gaps = pd.read_excel(PHASE2_XLSX, sheet_name="Gaps", engine="openpyxl")
df_rec  = pd.read_excel(PHASE2_XLSX, sheet_name="Recommendations", engine="openpyxl")

print("Loaded:", df_exec.shape, df_cov.shape, df_ev.shape, df_gaps.shape, df_rec.shape)

# Presentation style
sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams["figure.dpi"] = 140


# In[4]:


def build_scorecard(df_cov, df_ev, df_gaps):
    # Evidence summary
    sdg_tot = df_ev.groupby("SDG").agg(
        Evidence=("Sentence", "count"),
        AvgSim=("Semantic_Similarity", "mean"),
        AvgImpact=("Impact_Score", "mean"),
        Pages=("Page", lambda x: len(set(x)))
    ).reset_index()

    # Strength distribution
    cov_dist = df_cov.pivot_table(
        index="SDG", columns="Strength",
        values="Target_Indicator", aggfunc="count", fill_value=0
    )
    for col in ["Strong", "Moderate", "Weak", "None"]:
        if col not in cov_dist.columns:
            cov_dist[col] = 0
    cov_dist = cov_dist[["Strong", "Moderate", "Weak", "None"]].reset_index()

    # Gap counts
    gap_counts = df_gaps.groupby("SDG").size().rename("Gap_Count").reset_index()

    # Merge
    score_df = (sdg_tot.merge(cov_dist, on="SDG", how="left")
                      .merge(gap_counts, on="SDG", how="left")
                      .fillna({"Gap_Count": 0}))

    # Coverage Index
    weights = {"Strong": 1.0, "Moderate": 0.7, "Weak": 0.4, "None": 0.0}
    score_df["Targets_Total"] = score_df[["Strong","Moderate","Weak","None"]].sum(axis=1)
    score_df["Coverage_Index"] = (
        score_df["Strong"]*weights["Strong"] +
        score_df["Moderate"]*weights["Moderate"] +
        score_df["Weak"]*weights["Weak"] +
        score_df["None"]*weights["None"]
    ) / score_df["Targets_Total"].replace(0, np.nan)

    # Evidence Quality Index (normalized; adjust bounds if needed)
    score_df["Evidence_Quality_Index"] = ((score_df["AvgSim"] - 0.20) / (0.55 - 0.20)).clip(0, 1)

    # Gap Index (normalize by max gaps)
    max_g = max(score_df["Gap_Count"].max(), 1)
    score_df["Gap_Index"] = (score_df["Gap_Count"] / max_g).clip(0, 1)

    # Overall Score
    score_df["Overall_Score"] = (
        0.50*score_df["Coverage_Index"] +
        0.35*score_df["Evidence_Quality_Index"] +
        0.15*(1 - score_df["Gap_Index"])
    ) * 100

    # Rating
    bins = [0, 50, 65, 80, 100]
    labels = ["D", "C", "B", "A"]
    score_df["Rating"] = pd.cut(score_df["Overall_Score"], bins=bins, labels=labels, include_lowest=True)

    return score_df.sort_values("Overall_Score", ascending=False)

scorecard = build_scorecard(df_cov, df_ev, df_gaps)
display(scorecard)


# In[5]:


def savefig(name):
    p = OUT_DIR / f"{name}.png"
    plt.tight_layout()
    plt.savefig(p, bbox_inches="tight")
    print("Saved:", p)

order = sorted(df_cov["SDG"].unique())

dist = (df_cov.groupby(["SDG","Strength"]).size()
            .reset_index(name="Count")
            .pivot(index="SDG", columns="Strength", values="Count")
            .fillna(0)
       )

for col in ["Strong","Moderate","Weak","None"]:
    if col not in dist.columns:
        dist[col] = 0
dist = dist[["Strong","Moderate","Weak","None"]].loc[order]

colors = {"Strong":"#2ca02c", "Moderate":"#ff7f0e", "Weak":"#d62728", "None":"#7f7f7f"}

ax = dist.plot(kind="bar", stacked=True, figsize=(10,5),
               color=[colors[c] for c in dist.columns])

ax.set_title("SDG Target Coverage Strength Distribution")
ax.set_xlabel("SDG")
ax.set_ylabel("Number of Targets")
ax.legend(title="Strength", bbox_to_anchor=(1.02, 1), loc="upper left")

savefig("01_strength_distribution")
plt.show()


# In[7]:


top_targets = df_cov.sort_values("Evidence_Count", ascending=False).head(12)

plt.figure(figsize=(11,5))
sns.barplot(data=top_targets, x="Evidence_Count", y="Target_Indicator", hue="SDG", dodge=False)
plt.title("Top SDG Targets by Evidence Count")
plt.xlabel("Evidence Count")
plt.ylabel("Target / Indicator")
savefig("02_top_targets_by_evidence")
plt.show()


# In[8]:


heat = df_cov.pivot_table(index="SDG", columns="Target_Indicator",
                          values="Evidence_Count", fill_value=0)

plt.figure(figsize=(16,4))
sns.heatmap(heat, cmap="Blues", linewidths=0.3, linecolor="white")
plt.title("Heatmap: Evidence Count by SDG Target/Indicator")
plt.xlabel("Target/Indicator")
plt.ylabel("SDG")
savefig("03_heatmap_target_coverage")
plt.show()


# In[9]:


plt.figure(figsize=(10,5))
sns.boxplot(data=df_ev, x="SDG", y="Semantic_Similarity", order=order)
plt.title("Semantic Similarity Distribution by SDG (Evidence Quality)")
plt.xlabel("SDG")
plt.ylabel("Semantic Similarity")
savefig("04_similarity_boxplot")
plt.show()


# In[10]:


gap_counts = df_gaps.groupby("SDG").size().reset_index(name="Gap_Count")

plt.figure(figsize=(9,4))
sns.barplot(data=gap_counts, x="SDG", y="Gap_Count", order=order, color="#d62728")
plt.title("Number of Reporting Gaps Identified (Phase-2)")
plt.xlabel("SDG")
plt.ylabel("Gap Count")
savefig("05_gap_count")
plt.show()


# In[11]:


rec_mix = df_rec.groupby(["SDG","Type"]).size().reset_index(name="Count")

plt.figure(figsize=(10,5))
sns.barplot(data=rec_mix, x="SDG", y="Count", hue="Type", order=order)
plt.title("Recommendation Mix by SDG")
plt.xlabel("SDG")
plt.ylabel("Count")
plt.legend(title="Type", bbox_to_anchor=(1.02, 1), loc="upper left")
savefig("06_recommendation_mix")
plt.show()


# In[12]:


plt.figure(figsize=(10,5))
ax = sns.barplot(data=scorecard, x="SDG", y="Overall_Score", order=scorecard["SDG"].tolist(), palette="viridis")

for i, r in scorecard.reset_index(drop=True).iterrows():
    ax.text(i, r["Overall_Score"] + 1, f"{r['Rating']} ({r['Overall_Score']:.1f})",
            ha="center", va="bottom", fontsize=9)

plt.title("SDG Reporting Score (Coverage + Evidence Quality - Gap Penalty)")
plt.xlabel("SDG")
plt.ylabel("Score (0–100)")
plt.ylim(0, 100)
savefig("07_scorecard")
plt.show()


# In[13]:


import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

PHASE2_XLSX = Path("/Users/abc/Downloads/IPS/SDG_Output/SDG_Phase2_GapAnalysis_001.xlsx")

df_cov  = pd.read_excel(PHASE2_XLSX, sheet_name="Target_Coverage", engine="openpyxl")
df_ev   = pd.read_excel(PHASE2_XLSX, sheet_name="Evidence_All", engine="openpyxl")
df_gaps = pd.read_excel(PHASE2_XLSX, sheet_name="Gaps", engine="openpyxl")

def build_scorecard(df_cov, df_ev, df_gaps):
    # Evidence summary
    sdg_tot = df_ev.groupby("SDG").agg(
        Evidence=("Sentence", "count"),
        AvgSim=("Semantic_Similarity", "mean"),
        AvgImpact=("Impact_Score", "mean"),
        Pages=("Page", lambda x: len(set(x)))
    ).reset_index()

    # Strength distribution
    cov_dist = df_cov.pivot_table(
        index="SDG", columns="Strength",
        values="Target_Indicator", aggfunc="count", fill_value=0
    )
    for col in ["Strong", "Moderate", "Weak", "None"]:
        if col not in cov_dist.columns:
            cov_dist[col] = 0
    cov_dist = cov_dist[["Strong", "Moderate", "Weak", "None"]].reset_index()

    # Gap counts
    gap_counts = df_gaps.groupby("SDG").size().rename("Gap_Count").reset_index()

    # Merge
    score_df = (sdg_tot.merge(cov_dist, on="SDG", how="left")
                      .merge(gap_counts, on="SDG", how="left")
                      .fillna({"Gap_Count": 0}))

    # Coverage Index
    weights = {"Strong": 1.0, "Moderate": 0.7, "Weak": 0.4, "None": 0.0}
    score_df["Targets_Total"] = score_df[["Strong","Moderate","Weak","None"]].sum(axis=1).replace(0, np.nan)
    score_df["Coverage_Index"] = (
        score_df["Strong"]*weights["Strong"] +
        score_df["Moderate"]*weights["Moderate"] +
        score_df["Weak"]*weights["Weak"] +
        score_df["None"]*weights["None"]
    ) / score_df["Targets_Total"]

    # Evidence Quality Index (normalize semantic similarity)
    score_df["Evidence_Quality_Index"] = ((score_df["AvgSim"] - 0.20) / (0.55 - 0.20)).clip(0, 1)

    # Gap Control Index (higher = better control; based on fewer gaps)
    max_g = max(score_df["Gap_Count"].max(), 1)
    score_df["Gap_Penalty_Index"] = (score_df["Gap_Count"] / max_g).clip(0, 1)
    score_df["Gap_Control_Index"] = 1 - score_df["Gap_Penalty_Index"]

    # Overall Score (0–100)
    score_df["Overall_Score"] = (
        0.50*score_df["Coverage_Index"] +
        0.35*score_df["Evidence_Quality_Index"] +
        0.15*score_df["Gap_Control_Index"]
    ) * 100

    # Rating bands
    bins = [0, 50, 65, 80, 100]
    labels = ["D", "C", "B", "A"]
    score_df["Rating"] = pd.cut(score_df["Overall_Score"], bins=bins, labels=labels, include_lowest=True)

    return score_df.sort_values("Overall_Score", ascending=False)

scorecard = build_scorecard(df_cov, df_ev, df_gaps)
display(scorecard[["SDG","Overall_Score","Rating","Coverage_Index","Evidence_Quality_Index","Gap_Control_Index","Evidence","Gap_Count"]])


# In[14]:


plt.figure(figsize=(10,5))
order = scorecard["SDG"].tolist()

bars = plt.bar(order, scorecard["Overall_Score"], color="#2E86AB")
plt.ylim(0, 100)
plt.title("SDG Reporting Scorecard (Overall Score + Rating)")
plt.ylabel("Score (0–100)")
plt.xlabel("SDG")

for i, (s, r) in enumerate(zip(scorecard["Overall_Score"], scorecard["Rating"])):
    plt.text(i, s + 1, f"{r} ({s:.1f})", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.show()


# In[16]:


import numpy as np

def radar_for_one(ax, title, values, labels, color="#FF7F0E"):
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    values = values + values[:1]
    angles = angles + angles[:1]

    ax.plot(angles, values, color=color, linewidth=2)
    ax.fill(angles, values, color=color, alpha=0.15)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=11, pad=14)

labels = ["Coverage", "Evidence Quality", "Gap Control"]

rows = scorecard.copy()
rows["Coverage"] = rows["Coverage_Index"]
rows["Evidence Quality"] = rows["Evidence_Quality_Index"]
rows["Gap Control"] = rows["Gap_Control_Index"]

n = len(rows)
cols = 3
rws = int(np.ceil(n/cols))

fig, axes = plt.subplots(rws, cols, subplot_kw=dict(polar=True), figsize=(12, 4*rws))
axes = np.array(axes).reshape(-1)

for i, (_, row) in enumerate(rows.iterrows()):
    vals = [row["Coverage"], row["Evidence Quality"], row["Gap Control"]]
    radar_for_one(axes[i], f"{row['SDG']} | Rating {row['Rating']}", vals, labels)

# Hide unused axes
for j in range(i+1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()


# In[17]:


labels = ["Coverage_Index", "Evidence_Quality_Index", "Gap_Control_Index"]
pretty = ["Coverage", "Evidence Quality", "Gap Control"]
angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

plt.figure(figsize=(7,7))
ax = plt.subplot(111, polar=True)

colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#9467bd"]

for c, (_, row) in zip(colors, scorecard.iterrows()):
    vals = [row[l] for l in labels]
    vals += vals[:1]
    ax.plot(angles, vals, color=c, linewidth=2, label=f"{row['SDG']} ({row['Rating']})")
    ax.fill(angles, vals, color=c, alpha=0.08)

ax.set_thetagrids(np.degrees(angles[:-1]), pretty)
ax.set_ylim(0, 1)
ax.set_title("Drivers of SDG Reporting Score (Radar)", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10))
plt.tight_layout()
plt.show()


# In[18]:


import pandas as pd
from pathlib import Path

PHASE2_XLSX = Path("/Users/abc/Downloads/IPS/SDG_Output/SDG_Phase2_GapAnalysis_001.xlsx")

df_gaps = pd.read_excel(PHASE2_XLSX, sheet_name="Gaps", engine="openpyxl")
df_rec  = pd.read_excel(PHASE2_XLSX, sheet_name="Recommendations", engine="openpyxl")

top5_gaps = df_gaps.head(5).copy()
top5_recs = df_rec.head(5).copy()

print("Top 5 Gaps:")
display(top5_gaps)

print("Top 5 Recommendations:")
display(top5_recs)


# In[19]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10,4))

gap_counts = top5_gaps.groupby("SDG").size().reset_index(name="Count")
sns.barplot(data=gap_counts, x="SDG", y="Count", color="#d62728")

plt.title("Top 5 Gaps – Distribution by SDG")
plt.xlabel("SDG")
plt.ylabel("Count (in Top 5)")
plt.tight_layout()
plt.show()


# In[20]:


plt.figure(figsize=(10,4))

rec_counts = top5_recs.groupby("Type").size().reset_index(name="Count")
sns.barplot(data=rec_counts, x="Type", y="Count", palette="Set2")

plt.title("Top 5 Recommendations – Mix by Type")
plt.xlabel("Recommendation Type")
plt.ylabel("Count (in Top 5)")
plt.tight_layout()
plt.show()


# In[21]:


plt.figure(figsize=(12,5))
sns.barplot(
    data=top5_gaps.assign(Gap_Label=lambda d: d["SDG"] + " | " + d["gap"].str.slice(0, 55) + "..."),
    x=[1]*len(top5_gaps),
    y="Gap_Label",
    color="#d62728"
)
plt.title("Top 5 Gaps (CMD Summary)")
plt.xlabel("") 
plt.ylabel("")
plt.xticks([])
plt.tight_layout()
plt.show()


# In[ ]:




