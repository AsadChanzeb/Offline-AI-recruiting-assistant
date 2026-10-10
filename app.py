"""CVLens — Local CV matching engine.

Hybrid HNSW + BM25 + KD-Tree + Brute-Force search over Ollama embeddings,
context-preserved 250-word overlapping chunks, and a RAG pipeline with a
local LLM. All search algorithms are implemented from scratch.
"""

import heapq
import json
import logging
import math
import os
import random
import re
import threading
import time
from collections import Counter
from typing import Optional

import numpy as np
import requests
from flask import Flask, jsonify, request, send_file

from profiles import analyze, find_skills

# ---------------------------------------------------------------------------
# Configuration (override via environment variables)
# ---------------------------------------------------------------------------
BASE: str = os.path.dirname(os.path.abspath(__file__))
DB: str = os.path.join(BASE, "cvlens_db.json")
OLLAMA: str = os.environ.get("CVLENS_OLLAMA_URL", "http://127.0.0.1:11434")
EMB: str = os.environ.get("CVLENS_EMBED_MODEL", "nomic-embed-text")
GEN: str = os.environ.get("CVLENS_GEN_MODEL", "llama3.2")
CHUNK_WORDS: int = 250
OVERLAP: int = 50
OFFLINE: str = f"Ollama is offline or {EMB} is missing. Run: ollama pull {EMB}"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
lock = threading.Lock()
chunks: list[dict] = []   # chunks[i] = {cv: int, text: str, alive: bool}
vecs: list[np.ndarray] = []
cvs: dict[int, str] = {}
next_cv: int = 1
kd_dirty: bool = True
_mat: Optional[tuple[list[int], Optional[np.ndarray]]] = None

# ---------------------------------------------------------------------------
# BM25 In-Memory Keyword Indexer
# ---------------------------------------------------------------------------
class BM25Index:
    """Okapi BM25 ranking function with Lucene-style IDF.

    Maintains an in-memory inverted index over alive chunks and scores
    queries using TF-IDF with document-length normalization.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_lens: list[int] = []
        self.avg_len: float = 0.0
        self.doc_freqs: list[Counter] = []
        self.idf: dict[str, float] = {}
        self.live_ids: list[int] = []

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Split text into lowercase alphanumeric tokens."""
        return re.findall(r'[a-zA-Z0-9_\-\.]+', text.lower())

    def rebuild(self, alive_indices: list[int]) -> None:
        """Reconstruct the index from the given alive chunk indices."""
        self.live_ids = alive_indices
        self.doc_lens = []
        self.doc_freqs = []
        df: Counter = Counter()
        total_len = 0

        for i in self.live_ids:
            toks = self.tokenize(chunks[i]["text"])
            self.doc_lens.append(len(toks))
            total_len += len(toks)
            tf = Counter(toks)
            self.doc_freqs.append(tf)
            for word in tf.keys():
                df[word] += 1

        n_docs = len(self.live_ids)
        self.avg_len = (total_len / n_docs) if n_docs > 0 else 1.0
        self.idf = {}
        for word, freq in df.items():
            # Standard Lucene-style IDF
            self.idf[word] = math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))

    def search(self, query: str, top_k: int = 20) -> list[tuple[float, int]]:
        """Return up to *top_k* ``(score, chunk_index)`` pairs ranked by BM25."""
        if not self.live_ids:
            return []
        tokens = self.tokenize(query)
        scores: list[tuple[float, int]] = []
        for idx, (i, tf, d_len) in enumerate(zip(self.live_ids, self.doc_freqs, self.doc_lens)):
            score = 0.0
            for t in tokens:
                if t in tf:
                    freq = tf[t]
                    num = freq * (self.k1 + 1.0)
                    den = freq + self.k1 * (1.0 - self.b + self.b * (d_len / self.avg_len))
                    score += self.idf.get(t, 0.0) * (num / den)
            if score > 0:
                scores.append((score, i))
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:top_k]

bm25 = BM25Index()


def l2(a: np.ndarray, b: np.ndarray) -> float:
    """Squared L2 distance. For unit vectors: ``cos_sim = 1 - d/2``."""
    d = a - b
    return float(d @ d)


def pct(d: float) -> float:
    """Convert squared L2 distance to a 0–100 % similarity score."""
    return round((1 - d / 2) * 1000) / 10


# ---------------------------------------------------------------------------
# 1. Brute Force — exact, scans everything via matrix multiply
# ---------------------------------------------------------------------------
def matrix() -> tuple[list[int], Optional[np.ndarray]]:
    """Lazy-build a dense matrix of all alive vectors for brute-force search."""
    global _mat
    if _mat is None:
        ids = [i for i, c in enumerate(chunks) if c["alive"]]
        _mat = (ids, np.array([vecs[i] for i in ids]) if ids else None)
    return _mat


def brute(q: np.ndarray, k: int) -> list[tuple[float, int]]:
    """Exact nearest-neighbour search via a single matrix–vector dot product."""
    ids, M = matrix()
    if not ids:
        return []
    d = 2 - 2 * (M @ q)
    return [(float(d[j]), ids[j]) for j in np.argsort(d)[:k]]


# ---------------------------------------------------------------------------
# 2. KD-Tree — exact NN via axis-aligned binary space partitioning
# ---------------------------------------------------------------------------
class KD:
    """KD-Tree for exact nearest-neighbour search.

    Nodes are stored as ``(vec_index, axis, left_child, right_child)``
    tuples. The tree is rebuilt lazily when the corpus changes.
    """

    root = None

    def build(self, ids: list[int], dep: int):
        """Recursively build the tree by median-splitting on alternating axes."""
        if not ids:
            return None
        ax = dep % len(vecs[ids[0]])
        ids = sorted(ids, key=lambda i: vecs[i][ax])
        m = len(ids) // 2
        return (ids[m], ax, self.build(ids[:m], dep + 1), self.build(ids[m + 1:], dep + 1))

    def search(self, q: np.ndarray, k: int) -> list[tuple[float, int]]:
        """Find the *k* nearest neighbours, pruning branches by hyperplane distance."""
        global kd_dirty
        if kd_dirty:
            self.root = self.build([i for i, c in enumerate(chunks) if c["alive"]], 0)
            kd_dirty = False
        h: list[tuple[float, int]] = []

        def go(n):
            if n is None:
                return
            i, ax, left, right = n
            d = l2(q, vecs[i])
            if len(h) < k:
                heapq.heappush(h, (-d, i))
            elif d < -h[0][0]:
                heapq.heapreplace(h, (-d, i))
            diff = q[ax] - vecs[i][ax]
            near, far = (left, right) if diff < 0 else (right, left)
            go(near)
            if len(h) < k or diff * diff < -h[0][0]:
                go(far)

        go(self.root)
        return sorted((-d, i) for d, i in h)


# ---------------------------------------------------------------------------
# 3. HNSW — approximate multilayer navigable small-world graph
# ---------------------------------------------------------------------------
class HNSW:
    """Hierarchical Navigable Small World graph for approximate NN search.

    Follows the Malkov & Yashunin (2018) algorithm with ``M=16`` bidirectional
    links per node and ``efConstruction=100``.
    """

    M: int = 16        # max neighbours per layer
    EFC: int = 100     # ef during construction

    def __init__(self) -> None:
        self.nb: list[list[list[int]]] = []     # nb[node][layer] -> neighbour list
        self.entry: int = -1                    # entry-point node
        self.top: int = -1                      # highest occupied layer
        self.rng = random.Random(42)

    def layer(self, q: np.ndarray, ep: int, ef: int, l: int) -> list[tuple[float, int]]:
        """Greedy search on layer *l* starting from entry-point *ep*."""
        seen = {ep}
        d = l2(q, vecs[ep])
        cand, best = [(d, ep)], [(-d, ep)]
        while cand:
            cd, c = heapq.heappop(cand)
            if len(best) >= ef and cd > -best[0][0]:
                break
            for n in self.nb[c][l]:
                if n in seen:
                    continue
                seen.add(n)
                nd = l2(q, vecs[n])
                if len(best) < ef or nd < -best[0][0]:
                    heapq.heappush(cand, (nd, n))
                    heapq.heappush(best, (-nd, n))
                    if len(best) > ef:
                        heapq.heappop(best)
        return sorted((-d, i) for d, i in best)

    def add(self, i: int) -> None:
        """Insert vector *i* into the graph, connecting it on each layer."""
        L = int(-math.log(1 - self.rng.random()) / math.log(self.M))
        self.nb.append([[] for _ in range(L + 1)])
        if self.entry < 0:
            self.entry, self.top = i, L
            return
        q, ep = vecs[i], self.entry
        for l in range(self.top, L, -1):
            ep = self.layer(q, ep, 1, l)[0][1]
        for l in range(min(L, self.top), -1, -1):
            r = self.layer(q, ep, self.EFC, l)
            cap = 2 * self.M if l == 0 else self.M
            for _, o in r[:self.M]:
                self.nb[i][l].append(o)
                self.nb[o][l].append(i)
                if len(self.nb[o][l]) > cap:
                    self.nb[o][l] = sorted(self.nb[o][l], key=lambda x: l2(vecs[o], vecs[x]))[:cap]
            ep = r[0][1]
        if L > self.top:
            self.entry, self.top = i, L

    def search(self, q: np.ndarray, k: int, ef: int = 64) -> list[tuple[float, int]]:
        """Search the graph for *k* approximate nearest neighbours."""
        if self.entry < 0:
            return []
        ep = self.entry
        for l in range(self.top, 0, -1):
            ep = self.layer(q, ep, 1, l)[0][1]
        return [(d, i) for d, i in self.layer(q, ep, max(ef, k * 3), 0) if chunks[i]["alive"]][:k]


H, K = HNSW(), KD()


def run(algo: str, q: np.ndarray, k: int) -> list[tuple[float, int]]:
    """Dispatch a search query to the chosen algorithm."""
    return brute(q, k) if algo == "brute" else K.search(q, k) if algo == "kd" else H.search(q, k)


# ---------------------------------------------------------------------------
# Hybrid RRF Retrieval (Dense + Sparse BM25)
# ---------------------------------------------------------------------------
def hybrid_search(query_text: str, query_vec: np.ndarray, top_k: int = 8) -> list[tuple[float, int]]:
    """Fuse HNSW dense search with BM25 keyword matching via Reciprocal Rank Fusion.

    Each retriever independently ranks chunks, then their ranks are merged
    using RRF with ``k=60`` so that a chunk appearing in both lists is
    boosted while one appearing in only one list still contributes.
    """
    dense_hits = H.search(query_vec, max(30, top_k * 4))
    sparse_hits = bm25.search(query_text, top_k=30)

    RRF_CONST = 60
    rrf_scores: dict[int, float] = {}

    # 1. Score dense hits
    for rank, (dist, idx) in enumerate(dense_hits):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (RRF_CONST + rank + 1))

    # 2. Score sparse BM25 hits
    for rank, (score, idx) in enumerate(sparse_hits):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (RRF_CONST + rank + 1))

    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    # Compute L2 distance for the score presentation
    results: list[tuple[float, int]] = []
    for idx in sorted_ids[:top_k]:
        d = l2(query_vec, vecs[idx])
        results.append((d, idx))
    return results


# ---------------------------------------------------------------------------
# Ollama client helpers
# ---------------------------------------------------------------------------
def embed(text: str, prefix: str) -> Optional[np.ndarray]:
    """Embed *text* with ``nomic-embed-text`` via Ollama, returning a unit vector."""
    try:
        r = requests.post(OLLAMA + "/api/embeddings", json={"model": EMB, "prompt": prefix + text}, timeout=300)
        v = np.array(r.json()["embedding"], dtype=np.float32) if r.ok else None
    except Exception:
        return None
    if v is None:
        return None
    n = np.linalg.norm(v)
    return v / n if n else None


def generate(prompt: str) -> str:
    """Send a prompt to the local LLM and return its response."""
    try:
        r = requests.post(OLLAMA + "/api/generate", json={"model": GEN, "prompt": prompt, "stream": False}, timeout=600)
        return r.json().get("response", "") if r.ok else ""
    except Exception:
        return ""


def chunk_text(text: str, candidate_name: str = "") -> list[str]:
    """Split *text* into 250-word windows with 50-word overlap.

    Each chunk is optionally prefixed with the candidate's name so that
    the embedding model anchors entity information across pages.
    """
    words, out, step = text.split(), [], CHUNK_WORDS - OVERLAP
    for i in range(0, len(words), step):
        chunk_body = " ".join(words[i:i + CHUNK_WORDS])
        if candidate_name:
            full_chunk = f"Candidate: {candidate_name}\n{chunk_body} "
        else:
            full_chunk = chunk_body + " "
        out.append(full_chunk)
        if i + CHUNK_WORDS >= len(words):
            break
    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save() -> None:
    """Serialize all alive CVs and chunks to ``cvlens_db.json``."""
    live = [i for i, c in enumerate(chunks) if c["alive"]]
    data = {"cvs": [{"id": k, "name": v} for k, v in cvs.items()],
            "chunks": [{"cv": chunks[i]["cv"], "text": chunks[i]["text"], "v": vecs[i].tolist()} for i in live]}
    with open(DB, "w", encoding="utf-8") as f:
        json.dump(data, f)
    bm25.rebuild(live)


def load() -> None:
    """Load previously saved CVs and rebuild all search indices."""
    global next_cv
    if not os.path.exists(DB):
        return
    try:
        with open(DB, encoding="utf-8") as f:
            d = json.load(f)
        for c in d["cvs"]:
            cvs[c["id"]] = c["name"]
            next_cv = max(next_cv, c["id"] + 1)
        for c in d["chunks"]:
            chunks.append({"cv": c["cv"], "text": c["text"], "alive": True})
            vecs.append(np.array(c["v"], dtype=np.float32))
            H.add(len(chunks) - 1)
        bm25.rebuild([i for i, c in enumerate(chunks) if c["alive"]])
        log.info("Loaded %d CVs (%d chunks) from %s", len(cvs), len(chunks), DB)
    except Exception as e:
        log.error("Could not read %s: %s", DB, e)


def err(msg: str, code: int):
    """Return a JSON error response."""
    return jsonify(error=msg), code


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    """Serve the single-page UI."""
    return send_file(os.path.join(BASE, "index.html"))


@app.get("/status")
def status():
    up = e = g = False
    try:
        r = requests.get(OLLAMA + "/api/tags", timeout=3)
        if r.ok:
            up, names = True, r.text
            e, g = EMB in names, GEN in names
    except Exception:
        pass
    with lock:
        return jsonify(ollama=up, embed=e, gen=g, cvs=len(cvs), chunks=sum(c["alive"] for c in chunks))


@app.get("/cv/list")
def cv_list():
    with lock:
        return jsonify([{"id": i, "name": n, "chunks": sum(c["alive"] and c["cv"] == i for c in chunks)} for i, n in cvs.items()])


@app.post("/cv/add")
def cv_add():
    global next_cv, kd_dirty, _mat
    b = request.get_json(silent=True) or {}
    raw_name = b.get("name", "")
    
    # Infer candidate name from first line if not explicitly supplied
    if not raw_name:
        lines = [line.strip() for line in b.get("text", "").splitlines() if line.strip()]
        raw_name = lines[0][:40] if lines else f"Candidate {next_cv}"
        
    parts = chunk_text(b.get("text", ""), candidate_name=raw_name)
    if not parts:
        return err("no text found in this CV", 400)
    vs = []
    for p in parts:
        v = embed(p, "search_document: ")
        if v is None:
            return err(OFFLINE, 503)
        vs.append(v)
    with lock:
        cid, next_cv = next_cv, next_cv + 1
        cvs[cid] = raw_name
        for p, v in zip(parts, vs):
            chunks.append({"cv": cid, "text": p, "alive": True})
            vecs.append(v)
            H.add(len(chunks) - 1)
        kd_dirty, _mat = True, None
        save()
    return jsonify(id=cid, chunks=len(parts))


@app.delete("/cv/delete/<int:cid>")
def cv_delete(cid):
    global kd_dirty, _mat
    with lock:
        for c in chunks:
            if c["cv"] == cid:
                c["alive"] = False
        cvs.pop(cid, None)
        kd_dirty, _mat = True, None
        save()
    return jsonify(ok=True)


@app.post("/match")
def match():
    b = request.get_json(silent=True) or {}
    if "jd" not in b:
        return err("bad request", 400)
    q = embed(b["jd"], "search_query: ")
    if q is None:
        return err(OFFLINE, 503)
    algo, k = b.get("algo", "hnsw"), int(b.get("k", 8))
    with lock:
        t0 = time.perf_counter()
        hits = run(algo, q, max(40, k * 6))
        us = (time.perf_counter() - t0) * 1e6
        seen, out = set(), []
        for d, i in hits:
            cv = chunks[i]["cv"]
            if cv in seen:
                continue
            seen.add(cv)
            out.append({"cv": cv, "name": cvs.get(cv, ""), "score": pct(d), "snippet": chunks[i]["text"][:420]})
            if len(out) >= k:
                break
    return jsonify(results=out, algo=algo, us=us)


@app.post("/ask")
def ask():  # HYBRID RAG: Retrieve 8 chunks via HNSW + BM25 -> Llama 3.2
    b = request.get_json(silent=True) or {}
    question = b.get("question", "").strip()
    if not question:
        return err("bad request", 400)
        
    q = embed(question, "search_query: ")
    if q is None:
        return err(OFFLINE, 503)
        
    ctx, src = "", []
    with lock:
        # Fuses dense embeddings with sparse exact keyword matching across 8 chunks
        hybrid_hits = hybrid_search(question, q, top_k=8)
        for d, i in hybrid_hits:
            name = cvs.get(chunks[i]["cv"], "")
            ctx += f"[{name}]\n{chunks[i]['text']}\n\n"
            src.append({"name": name, "snippet": chunks[i]["text"][:350], "score": pct(d)})
            
    if not src:
        return err("Add at least one CV first.", 400)
        
    prompt = (
        "You are an expert technical recruiter assistant. Answer the user's question using ONLY the provided CV excerpts below.\n"
        "Rules:\n"
        "1. Mention the specific candidate name in [brackets] for each fact.\n"
        "2. If an exact hardware model, microcontroller part number, tool, or metric is mentioned in the excerpts, cite it explicitly.\n"
        "3. If the excerpts truly do not contain the answer, say 'The answer is not in the provided CV excerpts.'\n\n"
        f"CV EXCERPTS:\n{ctx}\n"
        f"QUESTION: {question}\n"
        "ANSWER:"
    )
    
    ans = generate(prompt)
    if not ans:
        return err("llama3.2 did not respond. Run: ollama pull llama3.2", 503)
    return jsonify(answer=ans, sources=src)


@app.post("/explain")
def explain():
    b = request.get_json(silent=True) or {}
    if "jd" not in b or "cv" not in b:
        return err("bad request", 400)
    with lock:
        cv_body = "\n".join(c["text"] for c in chunks if c["alive"] and c["cv"] == b["cv"])[:4000]
        name = cvs.get(b["cv"], "")
    ans = generate(f"You are a recruiter's assistant. JOB DESCRIPTION:\n{b['jd']}\n\nCV of {name}:\n{cv_body}\n\n"
                   "In 3 short bullets explain why this candidate matches the job, quoting the exact CV words as evidence. "
                   "Then add one line starting 'Gap:' naming the biggest missing requirement. Do not invent facts.")
    if not ans:
        return err("llama3.2 did not respond. Run: ollama pull llama3.2", 503)
    return jsonify(text=ans)


@app.post("/benchmark")
def benchmark():
    b = request.get_json(silent=True) or {}
    q = embed(b.get("text", ""), "search_query: ")
    if q is None:
        return err(OFFLINE, 503)
    with lock:
        truth = {i for _, i in brute(q, 10)}
        rows = []
        for a in ("brute", "kd", "hnsw"):
            run(a, q, 10)
            t0 = time.perf_counter()
            for _ in range(20):
                res = run(a, q, 10)
            us = (time.perf_counter() - t0) * 1e6 / 20
            rows.append({"algo": a, "us": us, "recall": len({i for _, i in res} & truth) / len(truth) if truth else 1.0})
        return jsonify(rows=rows, chunks=len(chunks))


def cv_text(cid: int) -> str:
    """Rebuild the full CV text from its overlapping chunks.

    Drops the ``Candidate: name`` prefix that was prepended during chunking
    and de-duplicates the overlap regions.
    """
    out: list[str] = []
    for c in chunks:
        if c["alive"] and c["cv"] == cid:
            t = c["text"]
            if t.startswith("Candidate: ") and "\n" in t:
                t = t.split("\n", 1)[1]
            w = t.split()
            out += w if not out else w[OVERLAP:]
    return " ".join(out)


@app.post("/candidates")
def candidates():  # structured profile per CV, plus skills asked for by the job description
    jd = find_skills((request.get_json(silent=True) or {}).get("jd", ""))
    with lock:
        out = []
        for cid, name in cvs.items():
            p = analyze(cv_text(cid)); p.update(id=cid, name=name, matched=[x for x in p["skills"] if x in jd]); out.append(p)
    return jsonify(candidates=out, jd_skills=jd)


@app.post("/cv/clear")
def cv_clear():
    global H, kd_dirty, _mat, next_cv
    with lock:
        chunks.clear()
        vecs.clear()
        cvs.clear()
        H = HNSW()
        kd_dirty = True
        _mat = None
        next_cv = 1
        bm25.rebuild([])
        save()
    return jsonify(ok=True)


load()
if __name__ == "__main__":
    log.info("CVLens running -> http://localhost:8080  (%d CVs loaded)", len(cvs))
    app.run("127.0.0.1", 8080, threaded=True)