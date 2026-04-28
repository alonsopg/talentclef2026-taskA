from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import json
import re

import pandas as pd


AUGMENTATION_VERSION = "deterministic_query_aug_v6"
AUGMENTATION_GENERATOR = "rule_based_structured_extraction"
DEFAULT_CACHE_ROOTNAME = "query_augmentation"
COMPACT_QUERY_MAX_WORDS = 28

_WS_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[^a-z0-9+/ #.-]+")
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|(?:\d+[\).\]])|(?:[a-zA-Z][\).\]])\s+)")
_EXPERIENCE_RE = re.compile(r"\b\d{1,2}\+?\s*(?:years?|anos|años)\b", re.I)

_HEADING_ALIASES = {
    "about the role": "role",
    "about role": "role",
    "role overview": "role",
    "job summary": "role",
    "summary": "role",
    "overview": "role",
    "about the position": "role",
    "about the job": "role",
    "about required skills": "skills",
    "required skills": "skills",
    "key skills": "skills",
    "core skills": "skills",
    "skills": "skills",
    "must have": "skills",
    "requirements": "requirements",
    "required qualifications": "requirements",
    "qualifications": "requirements",
    "education": "requirements",
    "responsibilities": "responsibilities",
    "key responsibilities": "responsibilities",
    "responsibilities key": "responsibilities",
    "main responsibilities": "responsibilities",
    "preferred qualifications": "preferred",
    "preferred skills": "preferred",
    "preferred requirements": "preferred",
    "nice to have": "preferred",
    "benefits": "other",
    "about us": "other",
    "introduction": "role",
    "acerca del puesto": "role",
    "acerca del rol": "role",
    "resumen del puesto": "role",
    "perfil del puesto": "role",
    "descripcion del puesto": "role",
    "descripción del puesto": "role",
    "habilidades requeridas": "skills",
    "competencias requeridas": "skills",
    "habilidades clave": "skills",
    "habilidades": "skills",
    "competencias": "skills",
    "responsabilidades clave": "responsibilities",
    "responsabilidades": "responsibilities",
    "funciones": "responsibilities",
    "requisitos": "requirements",
    "cualificaciones": "requirements",
    "formacion": "requirements",
    "formación": "requirements",
    "formacion requerida": "requirements",
    "formación requerida": "requirements",
    "cualificaciones preferidas": "preferred",
    "habilidades preferidas": "preferred",
    "deseable": "preferred",
    "buena a tener": "preferred",
}

_SENIORITY_PATTERNS = [
    (re.compile(r"\b(principal|director|head|vp|vice president|executive)\b", re.I), "principal"),
    (re.compile(r"\b(lead|senior|sr\.?)\b", re.I), "senior"),
    (re.compile(r"\b(manager|managerial)\b", re.I), "manager"),
    (re.compile(r"\b(mid[- ]level|intermediate)\b", re.I), "mid"),
    (re.compile(r"\b(associate|junior|jr\.?|entry level)\b", re.I), "junior"),
    (re.compile(r"\b(principal|director|directora?|jefe|jefa|ejecutiv[oa])\b", re.I), "principal"),
    (re.compile(r"\b(lider|líder|senior|s[eé]nior)\b", re.I), "senior"),
    (re.compile(r"\b(gerente|manager)\b", re.I), "manager"),
    (re.compile(r"\b(intermedio)\b", re.I), "mid"),
    (re.compile(r"\b(asociad[oa]|junior|jr\.?|nivel inicial)\b", re.I), "junior"),
]

_LANGUAGE_PATTERNS = {
    "english": re.compile(r"\benglish\b|\bingl[eé]s\b", re.I),
    "spanish": re.compile(r"\bspanish\b|\bespa[nñ]ol\b", re.I),
    "german": re.compile(r"\bgerman\b|\balem[aá]n\b", re.I),
    "french": re.compile(r"\bfrench\b|\bfranc[eé]s\b", re.I),
}

_EDUCATION_PATTERNS = [
    re.compile(r"\b(bachelor(?:'s)?|master(?:'s)?|phd|degree|diploma|mba)\b", re.I),
    re.compile(r"\b(grado|licenciatura|maestr[ií]a|doctorado|t[ií]tulo|diploma|mba)\b", re.I),
]

_TECH_HINT_RE = re.compile(
    r"\b("
    r"sql|python|excel|tableau|power bi|aws|azure|gcp|java|c\+\+|javascript|react|"
    r"node|sap|salesforce|jira|linux|docker|kubernetes|tensorflow|pytorch|sas|"
    r"matlab|cad|autocad|revit|crm|erp|api|etl|bi|hse|hvac"
    r")\b",
    re.I,
)

_TITLE_NORMALIZATION_PATTERNS = [
    re.compile(r"^(?:we are seeking|we're seeking|we seek|seeking)\s+(?:an?|the)\s+(.+?)\s+to join", re.I),
    re.compile(r"^(?:we are looking for|we're looking for)\s+(?:an?|the)\s+(.+?)\s+to join", re.I),
    re.compile(r"^(?:estamos buscando|buscamos)\s+(?:un/?a|una|un|el|la)?\s*(.+?)\s+para unirse", re.I),
    re.compile(r"^(?:estamos buscando|buscamos)\s+(?:un/?a|una|un|el|la)?\s*(.+?)\s+para incorporarse", re.I),
]

_LOW_SIGNAL_ITEMS = {
    "what you will do",
    "what we offer",
    "what we are looking for",
    "what you'll do",
    "lo que haras",
    "lo que harás",
    "lo que ofrecemos",
    "introduccion",
    "introducción",
    "introduction",
    "about the role",
    "about the job",
    "acerca del puesto",
    "acerca del rol",
}

_ITEM_PREFIX_PATTERNS = [
    re.compile(r"^(?:in this role you will|you will|you will be expected to)\s+", re.I),
    re.compile(r"^(?:en este puesto|como [^,.;:]+,|como [^,.;:]+)\s+", re.I),
]


def _normalize_ws(text: str) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    return _WS_RE.sub(" ", text).strip()


def _normalize_heading(text: str) -> str:
    text = _normalize_ws(text).lower().strip(":")
    text = _NON_WORD_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def _strip_bullet(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^\s*[-*•]\s*", "", text)
    text = re.sub(r"^\s*\d+[\).\]]\s*", "", text)
    text = re.sub(r"^\s*[a-zA-Z][\).\]]\s*", "", text)
    return _normalize_ws(text)


def _is_heading_line(line: str) -> Optional[str]:
    candidate = _normalize_heading(line)
    if not candidate:
        return None
    if candidate in _HEADING_ALIASES:
        return _HEADING_ALIASES[candidate]
    if len(candidate.split()) <= 4 and candidate.endswith("skills"):
        return "skills"
    if len(candidate.split()) <= 4 and candidate.endswith("responsibilities"):
        return "responsibilities"
    if len(candidate.split()) <= 4 and candidate.endswith("requirements"):
        return "requirements"
    return None


def _split_lines(text: str) -> List[str]:
    return [line.rstrip() for line in str(text or "").splitlines()]


def _clean_item(text: str) -> str:
    text = _strip_bullet(text)
    text = text.strip(" -:;,.")
    return _normalize_ws(text)


def _dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        item = _normalize_ws(item)
        if not item:
            continue
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _truncate_items(items: List[str], limit: int) -> List[str]:
    return items[:limit]


def _word_truncate(text: str, max_words: int) -> str:
    words = _normalize_ws(text).split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]).rstrip(" ,;:-") + "..."


def _pack_segments(segments: Iterable[str], max_words: int) -> str:
    words_left = int(max_words)
    packed: List[str] = []
    for segment in segments:
        segment = _normalize_ws(segment)
        if not segment:
            continue
        segment_words = segment.split()
        if not segment_words:
            continue
        if len(segment_words) <= words_left:
            packed.append(segment)
            words_left -= len(segment_words)
        elif words_left > 0:
            packed.append(_word_truncate(segment, words_left))
            words_left = 0
        if words_left <= 0:
            break
    return _normalize_ws(" ".join(packed))


def _clean_list_items(items: Iterable[str], max_items: int, max_words_per_item: int) -> List[str]:
    cleaned: List[str] = []
    for item in items:
        item = _normalize_ws(str(item or ""))
        if not item:
            continue
        for pattern in _ITEM_PREFIX_PATTERNS:
            item = pattern.sub("", item)
        normalized = _normalize_heading(item)
        if normalized in _LOW_SIGNAL_ITEMS:
            continue
        item = _word_truncate(item, max_words_per_item).strip(" .")
        if item:
            cleaned.append(item)
    return _truncate_items(_dedupe_keep_order(cleaned), max_items)


def normalize_title_candidate(title: str, raw_text: str) -> str:
    candidate = _normalize_ws(title)
    for pattern in _TITLE_NORMALIZATION_PATTERNS:
        match = pattern.search(candidate)
        if match:
            candidate = _normalize_ws(match.group(1))
            break

    if len(candidate.split()) > 14:
        first_line = _split_lines(raw_text)[0] if _split_lines(raw_text) else candidate
        first_line = _normalize_ws(first_line)
        if len(first_line.split()) <= 14:
            candidate = first_line

    return candidate.strip(" .:-")


def parse_job_description(raw_text: str) -> Dict[str, object]:
    lines = _split_lines(raw_text)
    title = ""
    preface = []
    sections: Dict[str, List[str]] = {
        "role": [],
        "skills": [],
        "responsibilities": [],
        "requirements": [],
        "preferred": [],
        "other": [],
    }
    current = "preface"

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if not title:
            title = _clean_item(stripped)
            continue

        heading = _is_heading_line(stripped)
        if heading is not None:
            current = heading
            continue

        target = preface if current == "preface" else sections[current]
        target.append(_clean_item(stripped))

    role_lines = sections["role"] or preface
    summary = " ".join(role_lines[:3]).strip()
    if not summary and preface:
        summary = " ".join(preface[:3]).strip()

    responsibilities = _dedupe_keep_order(
        item for item in sections["responsibilities"] if _clean_item(item)
    )

    skills = _dedupe_keep_order(
        item for item in sections["skills"] if _clean_item(item)
    )

    requirements = _dedupe_keep_order(
        item for item in sections["requirements"] if _clean_item(item)
    )

    preferred = _dedupe_keep_order(
        item for item in sections["preferred"] if _clean_item(item)
    )

    if not responsibilities:
        responsibilities = _dedupe_keep_order(
            item for item in role_lines if len(item.split()) >= 4
        )

    return {
        "title": title,
        "summary": summary,
        "role_lines": _dedupe_keep_order(role_lines),
        "skills": skills,
        "responsibilities": responsibilities,
        "requirements": requirements,
        "preferred": preferred,
        "other": _dedupe_keep_order(sections["other"]),
    }


def infer_seniority(text: str) -> str:
    text = str(text or "")
    for pattern, label in _SENIORITY_PATTERNS:
        if pattern.search(text):
            return label
    return ""


def extract_experience_terms(text: str) -> List[str]:
    return _dedupe_keep_order(match.group(0) for match in _EXPERIENCE_RE.finditer(str(text or "")))


def extract_language_terms(text: str) -> List[str]:
    found = []
    for label, pattern in _LANGUAGE_PATTERNS.items():
        if pattern.search(str(text or "")):
            found.append(label)
    return found


def extract_education_terms(text: str) -> List[str]:
    lines = _split_lines(text)
    hits: List[str] = []
    for line in lines:
        if any(pattern.search(line) for pattern in _EDUCATION_PATTERNS):
            hits.append(_clean_item(line))
    return _dedupe_keep_order(hits)


def extract_tool_terms(text: str, skills: Iterable[str]) -> List[str]:
    hints = [match.group(0).upper() for match in _TECH_HINT_RE.finditer(str(text or ""))]
    skill_hits = [item for item in skills if _TECH_HINT_RE.search(item)]
    return _dedupe_keep_order(hints + skill_hits)


def build_profile_text(aug: Dict[str, object], lang: str) -> str:
    title = aug.get("title", "")
    summary = aug.get("summary", "")
    seniority = aug.get("seniority", "")
    responsibilities = _truncate_items(list(aug.get("responsibilities", [])), 4)
    skills = _truncate_items(list(aug.get("skills", [])), 8)
    preferred = _truncate_items(list(aug.get("preferred", [])), 4)
    requirements = _truncate_items(list(aug.get("requirements", [])), 4)
    tools = _truncate_items(list(aug.get("tools", [])), 5)
    experience = _truncate_items(list(aug.get("experience_terms", [])), 3)
    education = _truncate_items(list(aug.get("education_terms", [])), 2)
    languages = _truncate_items(list(aug.get("language_terms", [])), 3)

    if lang == "es":
        parts = [
            f"Rol objetivo: {title}." if title else "",
            f"Nivel: {seniority}." if seniority else "",
            f"Resumen del puesto: {summary}." if summary else "",
            f"Responsabilidades clave: {'; '.join(responsibilities)}." if responsibilities else "",
            f"Habilidades requeridas: {'; '.join(skills)}." if skills else "",
            f"Habilidades deseables: {'; '.join(preferred)}." if preferred else "",
            f"Herramientas o tecnologias: {'; '.join(tools)}." if tools else "",
            f"Experiencia solicitada: {'; '.join(experience)}." if experience else "",
            f"Formacion esperada: {'; '.join(education)}." if education else "",
            f"Idiomas: {'; '.join(languages)}." if languages else "",
            f"Requisitos: {'; '.join(requirements)}." if requirements else "",
        ]
    else:
        parts = [
            f"Target role: {title}." if title else "",
            f"Seniority: {seniority}." if seniority else "",
            f"Role summary: {summary}." if summary else "",
            f"Key responsibilities: {'; '.join(responsibilities)}." if responsibilities else "",
            f"Required skills: {'; '.join(skills)}." if skills else "",
            f"Preferred skills: {'; '.join(preferred)}." if preferred else "",
            f"Tools or technologies: {'; '.join(tools)}." if tools else "",
            f"Requested experience: {'; '.join(experience)}." if experience else "",
            f"Expected education: {'; '.join(education)}." if education else "",
            f"Languages: {'; '.join(languages)}." if languages else "",
            f"Requirements: {'; '.join(requirements)}." if requirements else "",
        ]

    return _normalize_ws(" ".join(part for part in parts if part))


def build_ideal_resume_text(aug: Dict[str, object], lang: str) -> str:
    title = aug.get("title", "")
    seniority = aug.get("seniority", "")
    summary = aug.get("summary", "")
    skills = _truncate_items(list(aug.get("skills", [])), 8)
    responsibilities = _truncate_items(list(aug.get("responsibilities", [])), 4)
    experience = _truncate_items(list(aug.get("experience_terms", [])), 3)
    education = _truncate_items(list(aug.get("education_terms", [])), 2)
    languages = _truncate_items(list(aug.get("language_terms", [])), 3)

    if lang == "es":
        parts = [
            "Perfil ideal del candidato:",
            f"Profesional para el puesto de {title}." if title else "",
            f"Nivel esperado: {seniority}." if seniority else "",
            f"Experiencia demostrable en {'; '.join(responsibilities)}." if responsibilities else "",
            f"Competencias principales: {'; '.join(skills)}." if skills else "",
            f"Resumen del ajuste: {summary}." if summary else "",
            f"Trayectoria esperada: {'; '.join(experience)}." if experience else "",
            f"Formacion o credenciales: {'; '.join(education)}." if education else "",
            f"Idiomas relevantes: {'; '.join(languages)}." if languages else "",
        ]
    else:
        parts = [
            "Ideal candidate profile:",
            f"Professional aligned with the {title} role." if title else "",
            f"Expected seniority: {seniority}." if seniority else "",
            f"Demonstrated experience in {'; '.join(responsibilities)}." if responsibilities else "",
            f"Core competencies: {'; '.join(skills)}." if skills else "",
            f"Fit summary: {summary}." if summary else "",
            f"Expected background: {'; '.join(experience)}." if experience else "",
            f"Education or credentials: {'; '.join(education)}." if education else "",
            f"Relevant languages: {'; '.join(languages)}." if languages else "",
        ]

    return _normalize_ws(" ".join(part for part in parts if part))


def build_skills_text(skills: Iterable[str], tools: Iterable[str], lang: str) -> str:
    skills = _truncate_items(_dedupe_keep_order(list(skills) + list(tools)), 12)
    if not skills:
        return ""
    if lang == "es":
        return _normalize_ws(f"Habilidades y tecnologias clave: {'; '.join(skills)}.")
    return _normalize_ws(f"Key skills and technologies: {'; '.join(skills)}.")


def build_compact_rewrite_text(aug: Dict[str, object], lang: str, max_words: int = COMPACT_QUERY_MAX_WORDS) -> str:
    title = _word_truncate(_normalize_ws(aug.get("title", "")), 4).strip(".")
    seniority = _normalize_ws(aug.get("seniority", ""))
    skills = _clean_list_items(aug.get("skills", []), max_items=5, max_words_per_item=2)
    tools = _clean_list_items(aug.get("tools", []), max_items=2, max_words_per_item=2)
    languages = _clean_list_items(aug.get("language_terms", []), max_items=2, max_words_per_item=2)

    if lang == "es":
        segments = [
            f"Rol {title}." if title else "",
            f"Nivel {seniority}." if seniority else "",
            f"Habilidades {'; '.join(skills)}." if skills else "",
            f"Herramientas {'; '.join(tools)}." if tools else "",
            f"Idiomas {'; '.join(languages)}." if languages else "",
        ]
    else:
        segments = [
            f"Role {title}." if title else "",
            f"Seniority {seniority}." if seniority else "",
            f"Skills {'; '.join(skills)}." if skills else "",
            f"Tools {'; '.join(tools)}." if tools else "",
            f"Languages {'; '.join(languages)}." if languages else "",
        ]

    return _pack_segments(segments, max_words=max_words)


def augment_single_query(raw_text: str, qid: str, lang: str) -> Dict[str, object]:
    parsed = parse_job_description(raw_text)
    combined_text = _normalize_ws(raw_text)
    parsed["title"] = normalize_title_candidate(parsed.get("title", ""), raw_text)
    parsed["qid"] = str(qid)
    parsed["lang"] = str(lang)
    parsed["augmentation_version"] = AUGMENTATION_VERSION
    parsed["augmentation_generator"] = AUGMENTATION_GENERATOR
    parsed["source_sha256"] = sha256(str(raw_text or "").encode("utf-8")).hexdigest()
    parsed["seniority"] = infer_seniority(f"{parsed.get('title', '')} {combined_text}")
    parsed["experience_terms"] = extract_experience_terms(combined_text)
    parsed["education_terms"] = extract_education_terms(raw_text)
    parsed["language_terms"] = extract_language_terms(combined_text)
    parsed["tools"] = extract_tool_terms(combined_text, parsed.get("skills", []))
    parsed["profile_text"] = build_profile_text(parsed, lang)
    parsed["ideal_resume_text"] = build_ideal_resume_text(parsed, lang)
    parsed["skills_text"] = build_skills_text(parsed.get("skills", []), parsed.get("tools", []), lang)
    parsed["compact_rewrite_text"] = build_compact_rewrite_text(parsed, lang)

    base_query = _normalize_ws(raw_text)
    parsed["query_original"] = base_query
    parsed["query_profile"] = _normalize_ws(f"{base_query}\n\n{parsed['profile_text']}")
    parsed["query_ideal_resume"] = _normalize_ws(f"{base_query}\n\n{parsed['ideal_resume_text']}")
    parsed["query_profile_ideal_resume"] = _normalize_ws(
        f"{base_query}\n\n{parsed['profile_text']}\n\n{parsed['ideal_resume_text']}"
    )
    parsed["query_profile_skills"] = _normalize_ws(
        f"{base_query}\n\n{parsed['profile_text']}\n\n{parsed['skills_text']}"
    )
    parsed["query_profile_ideal_resume_skills"] = _normalize_ws(
        f"{base_query}\n\n{parsed['profile_text']}\n\n{parsed['ideal_resume_text']}\n\n{parsed['skills_text']}"
    )
    parsed["query_compact_rewrite"] = parsed["compact_rewrite_text"]
    return parsed


def build_cache_key(raw_text: str, lang: str) -> str:
    payload = "\n".join(
        [
            AUGMENTATION_VERSION,
            AUGMENTATION_GENERATOR,
            str(lang),
            sha256(str(raw_text or "").encode("utf-8")).hexdigest(),
        ]
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def cache_file_for_query(cache_root: Path, split_name: str, lang: str, qid: str) -> Path:
    return Path(cache_root) / split_name / lang / f"{qid}.json"


def save_augmentation_record(record: Dict[str, object], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")


def load_augmentation_record(path: Path) -> Dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def augment_topics(
    topics: pd.DataFrame,
    lang: str,
    split_name: str,
    cache_root: Path,
    overwrite: bool = False,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    manifest_rows: List[Dict[str, object]] = []
    cache_root = Path(cache_root)

    for row in topics[["qid", "raw_query", "query"]].itertuples(index=False):
        qid = str(row.qid)
        raw_query = str(row.raw_query)
        cache_path = cache_file_for_query(cache_root, split_name, lang, qid)
        expected_key = build_cache_key(raw_query, lang)

        use_cache = False
        if cache_path.exists() and not overwrite:
            cached = load_augmentation_record(cache_path)
            if cached.get("cache_key") == expected_key:
                use_cache = True
                record = cached
            else:
                record = augment_single_query(raw_query, qid=qid, lang=lang)
        else:
            record = augment_single_query(raw_query, qid=qid, lang=lang)

        if not use_cache:
            record["cache_key"] = expected_key
            save_augmentation_record(record, cache_path)

        manifest_rows.append(
            {
                "qid": qid,
                "lang": lang,
                "split": split_name,
                "cache_path": str(cache_path),
                "cache_key": expected_key,
                "augmentation_version": AUGMENTATION_VERSION,
                "augmentation_generator": AUGMENTATION_GENERATOR,
            }
        )
        rows.append(record)

    manifest_path = Path(cache_root) / split_name / lang / "_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_payload = {
        "split": split_name,
        "lang": lang,
        "augmentation_version": AUGMENTATION_VERSION,
        "augmentation_generator": AUGMENTATION_GENERATOR,
        "num_queries": len(rows),
        "queries": manifest_rows,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    aug_df = pd.DataFrame(rows)
    base = topics.copy()
    base["qid"] = base["qid"].astype(str)
    aug_df["qid"] = aug_df["qid"].astype(str)
    merged = base.merge(aug_df, on="qid", how="left", suffixes=("", "_aug"))
    return merged


def build_query_view(topics_aug: pd.DataFrame, view_name: str) -> pd.DataFrame:
    view_map = {
        "ORIGINAL": "query",
        "PROFILE": "query_profile",
        "IDEAL_RESUME": "query_ideal_resume",
        "PROFILE_IDEAL_RESUME": "query_profile_ideal_resume",
        "PROFILE_SKILLS": "query_profile_skills",
        "PROFILE_IDEAL_RESUME_SKILLS": "query_profile_ideal_resume_skills",
        "COMPACT_REWRITE": "query_compact_rewrite",
    }
    key = view_name.upper().strip()
    if key not in view_map:
        raise ValueError(f"Unsupported query view: {view_name}")
    column = view_map[key]
    out = topics_aug[["qid", column]].copy()
    out = out.rename(columns={column: "query"})
    out["qid"] = out["qid"].astype(str)
    out["query"] = out["query"].fillna("").astype(str)
    return out


def summarize_augmentation(topics_aug: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "qid",
        "title",
        "seniority",
        "experience_terms",
        "language_terms",
        "skills",
        "profile_text",
        "ideal_resume_text",
        "compact_rewrite_text",
    ]
    summary = topics_aug[columns].copy()
    for col in ["experience_terms", "language_terms", "skills"]:
        summary[col] = summary[col].apply(lambda x: ", ".join(x) if isinstance(x, list) else "")
    return summary
