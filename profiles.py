"""Rule-based CV profile extraction for CVLens.

Extracts years of experience, education level, research signals, and
technical skills from raw CV text using regex heuristics. Pure functions
with no dependency on app.py.

Usage::

    from profiles import analyze, find_skills

    profile = analyze(cv_text)   # -> dict with years, edu, skills, …
    jd_skills = find_skills(jd)  # -> list[str]
"""

import datetime
import re

MON = "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
RANGE = re.compile(rf"(?:\b({MON})[a-z]*\.?,?\s+)?\b((?:19|20)\d{{2}})\s*(?:-|\u2013|\u2014|to|until)\s*(?:\b({MON})[a-z]*\.?,?\s+)?\b(Present|Current|Now|Ongoing|Today|(?:19|20)\d{{2}})\b", re.I)
DEGREE_WORDS = re.compile(r"\b(bachelor\w*|master\w*|bs|ms|bsc|msc|b\.?sc|m\.?sc|mba|phd|degree|diploma|b\.?com|bba|bfa|intermediate|fsc|a-levels?|o-levels?|matric\w*)\b", re.I)
EDU = [(4, "PhD", r"\b(Ph\.?D\.?|Doctorate|Doctor of Philosophy)\b"),
       (3, "Master's", r"\b(M\.?Sc\.?|MS(?!\s*(?:Office|Excel|Word|Project|SQL|Access|Teams|Power|Azure))|MBA|M\.?Phil|M\.?Tech|M\.?Eng|Master(?:'s|s)?)\b"),
       (2, "Bachelor's", r"\b(B\.?Sc\.?|BS|BE|B\.?Tech|BBA|B\.?Com|BFA|BCS|BSCS|BSSE|Bachelor(?:'s|s)?)\b"),
       (1, "Diploma", r"\b(Diploma|Associate Degree|Intermediate|FSc|A-Levels?)\b")]
RESEARCH = {"published": r"\bpublish\w*|\bpublications?\b", "paper": r"\bpapers?\b", "journal": r"\bjournals?\b", "conference": r"\bconferences?\b",
            "thesis": r"\btheses|\bthesis\b|\bdissertation\b", "arXiv/IEEE": r"\barxiv\b|\bieee\b|\bacm\b", "patent": r"\bpatents?\b",
            "research role": r"\bresearch (?:assistant|associate|scientist|fellow|engineer|lab)\b|\bresearcher\b"}
SKILLS = {"Python": r"python", "Java": r"java", "C++": r"c\+\+", "C#": r"c#", "JavaScript": r"javascript", "TypeScript": r"typescript", "SQL": r"sql|postgres\w*|mysql",
          "React": r"react", "Node.js": r"node\.?js", "Django": r"django", "Flask": r"flask", "FastAPI": r"fastapi", "PyTorch": r"pytorch", "TensorFlow": r"tensorflow",
          "scikit-learn": r"scikit-learn|sklearn", "Pandas": r"pandas", "Hugging Face": r"hugging ?face|transformers", "LangChain": r"langchain|langgraph|llamaindex",
          "RAG": r"rag|retrieval[- ]augmented", "LLM": r"llms?|large language models?|language models?", "Fine-tuning": r"fine[- ]?tun\w*|lora|qlora|peft|dpo",
          "Embeddings": r"embedd?ings?", "Vector DB": r"vector (?:database|db|search|similarity)\w*|qdrant|chroma|faiss|pinecone|pgvector|weaviate|milvus|hnsw",
          "Ollama": r"ollama", "vLLM": r"vllm", "OpenAI API": r"openai|chatgpt|gpt-?\d*", "NLP": r"nlp|natural language processing|bert|spacy",
          "Computer Vision": r"computer vision|opencv|yolo", "MLOps": r"mlops|mlflow|kubeflow|kserve|triton", "Docker": r"docker", "Kubernetes": r"kubernetes|k8s",
          "AWS": r"aws|sagemaker", "Azure": r"azure", "GCP": r"gcp|google cloud", "Terraform": r"terraform", "CI/CD": r"ci/cd|github actions|jenkins", "Linux": r"linux",
          "Git": r"git", "MongoDB": r"mongodb", "Redis": r"redis", "Elasticsearch": r"elasticsearch", "Power BI": r"power ?bi", "Tableau": r"tableau", "Excel": r"excel",
          "Prompt Engineering": r"prompt (?:engineering|design)"}
SKILL_RE = {k: re.compile(r"(?<![A-Za-z0-9+#])(?:%s)(?![A-Za-z0-9+#])" % v, re.I) for k, v in SKILLS.items()}


def find_skills(text: str) -> list[str]:
    """Return a list of recognised skill names found in text."""
    return [k for k, r in SKILL_RE.items() if r.search(text)]


def analyze(text: str) -> dict:
    """Extract a structured profile from raw CV text.

    Returns a dict with keys:
        - ``years`` (float): estimated years of professional experience
        - ``edu`` (int): education level (0–4, higher = more advanced)
        - ``edu_label`` (str): human-readable degree name
        - ``edu_line`` (str): snippet of text where the degree was found
        - ``research`` (int): research signal strength score
        - ``research_tags`` (list[str]): matched research indicators
        - ``skills`` (list[str]): technical skills found in the text
    """
    now = datetime.date.today()
    intervals: list[tuple[int, int]] = []

    # Detect the education section boundary so we don't count study periods as work
    edu_m = re.search(r"\bEDUCATION\s*(?:&|AND)\s*CERTIFICATIONS?\b", text, re.I)
    edu_pos = edu_m.start() if edu_m else len(text)

    for m in RANGE.finditer(text):
        if m.start() >= edu_pos:
            continue  # inside education section, not work experience
        if DEGREE_WORDS.search(text[max(0, m.start() - 120):m.start()]):
            continue  # study period, not work
        mo = lambda g, d: (
            (MON.lower().split("|").index(g[:3].lower()) + 1)
            if g and g[:3].lower() in MON.lower().split("|")
            else d
        )
        y1, e = int(m.group(2)), m.group(4)
        a = y1 * 12 + mo(m.group(1), 6)
        b = now.year * 12 + now.month if e.isalpha() else int(e) * 12 + mo(m.group(3), 6)
        if 1980 <= y1 and a <= b <= now.year * 12 + now.month + 12:
            intervals.append((a, b))

    # Union of overlapping date ranges so concurrent jobs aren't double-counted
    intervals.sort()
    months, end = 0, None
    for a, b in intervals:
        if end is None or a > end:
            months += b - a
            end = b
        elif b > end:
            months += b - end
            end = b
    years = round(months / 12, 1)

    # Fallback: look for explicit "X years of experience" phrases
    if not years:
        st = [int(x) for x in re.findall(
            r"(\d{1,2})\+?\s+years?\s+of\s+(?:\w+\s+){0,3}experience", text, re.I
        )]
        years = float(max(st)) if st else 0.0

    # Highest education level
    lvl, label, line = 0, "Not found", ""
    for l, name, pat in EDU:
        m = re.search(pat, text)
        if m:
            lvl, label = l, name
            seg = text[m.start():m.start() + 70]
            y = re.search(r"(?:19|20)\d{2}", seg)
            line = seg[:y.end()] if y else seg[:max(seg.rfind(","), seg.rfind(" "))]
            break

    # Research signals
    tags = [k for k, p in RESEARCH.items() if re.search(p, text, re.I)]

    return dict(
        years=years,
        edu=lvl,
        edu_label=label,
        edu_line=line,
        research=len(tags) + (1 if "research role" in tags else 0),
        research_tags=tags,
        skills=find_skills(text),
    )
