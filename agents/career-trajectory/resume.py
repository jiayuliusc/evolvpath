"""Résumé ingestion: file/text extraction adapters, hygiene, and normalization.

Separation of concerns (each stage a plain function, all deterministic):

1. **Extraction** — ``extract_text`` / ``parse_resource`` turn raw bytes or a
   Chat Protocol ``Resource`` into plain text, via per-format adapters
   (txt/md/json natively, PDF via ``pypdf``, DOCX via ``python-docx``).
   Unsupported or undecodable formats return a structured
   ``{"status": "unsupported", ...}`` response — never a pretend parse.
2. **Hygiene** — ``sanitize`` strips control characters, caps length and
   detects instruction-like (prompt-injection) text; such text is ignored as
   content and reported as a warning. Résumé content is treated strictly as
   untrusted career data: nothing is executed, no embedded URLs are fetched,
   and raw text never reaches the LLM or logs (only extracted facts do).
3. **Fact extraction + normalization** — ``parse_resume`` delegates to
   ``profile._parse_resume_text``/``normalize_input`` so résumé text and pasted
   text share one parser, and returns a canonical-profile fragment with
   ``provenance="resume"`` plus parsing warnings.
"""

import base64
import binascii
import json
import re
from datetime import date

import career_profile as pf

MAX_FILE_BYTES = 2_000_000  # 2 MB cap on decoded résumé payloads

_TEXT_EXTENSIONS = (".txt", ".md", ".markdown", ".text", ".json", "")


class UnsupportedFormat(Exception):
    """Raised when no adapter can safely handle a payload."""


def _pdf_text(data: bytes) -> str:
    import io

    from pypdf import PdfReader  # imported lazily so tests can monkeypatch

    reader = PdfReader(io.BytesIO(data))
    pages = [(page.extract_text() or "") for page in reader.pages[:40]]
    return "\n".join(pages)


def _docx_text(data: bytes) -> str:
    import io

    from docx import Document  # python-docx, imported lazily

    document = Document(io.BytesIO(data))
    lines = [p.text for p in document.paragraphs[:600]]
    for table in document.tables[:20]:
        for row in table.rows[:50]:
            lines.append(" | ".join(cell.text for cell in row.cells[:10]))
    return "\n".join(lines)


def _extension_from_name(name: str) -> str:
    name = str(name or "").lower().rstrip('"')
    match = re.search(r"(\.[a-z0-9]{1,8})$", name)
    return match.group(1) if match else ""


def extract_text(filename: str, data: bytes) -> dict:
    """Extract plain text from résumé bytes. Returns
    ``{"status": "ok", "text": ...}`` or ``{"status": "unsupported"|"error", ...}``.

    Dispatch is by extension first, then content sniffing (PDF/DOCX magic
    bytes), so a mislabeled file still routes to the right adapter.
    """
    if not isinstance(data, bytes) or not data:
        return {"status": "error", "error": "empty file"}
    if len(data) > MAX_FILE_BYTES:
        return {"status": "error", "error": f"file exceeds the {MAX_FILE_BYTES // 1_000_000} MB limit"}
    extension = _extension_from_name(filename)
    sniff = data[:8]
    try:
        if extension == ".pdf" or sniff.startswith(b"%PDF"):
            return {"status": "ok", "text": _pdf_text(data), "format": "pdf"}
        if extension == ".docx" or sniff.startswith(b"PK\x03\x04"):
            return {"status": "ok", "text": _docx_text(data), "format": "docx"}
        if extension in (".doc",):  # legacy binary Word: no safe adapter
            return {"status": "unsupported",
                    "error": "legacy .doc is not supported; save as .docx or paste the text"}
        if extension not in _TEXT_EXTENSIONS:
            return {"status": "unsupported",
                    "error": f"'{extension or 'unknown'}' files are not supported; paste text or use PDF/DOCX/TXT/JSON"}
        text = data.decode("utf-8", errors="replace")
        if extension == ".json" or (extension == "" and text[:1] in "{["):
            try:
                json.loads(text)
            except ValueError:
                return {"status": "error", "error": "file looks like JSON but does not parse"}
        return {"status": "ok", "text": text, "format": extension.lstrip(".") or "text"}
    except UnsupportedFormat:
        raise
    except Exception as exc:  # adapter failure -> structured error, never a crash
        return {"status": "error", "error": f"could not extract text from {pf.redact(str(exc))[:80]}"}


def parse_resource(resource: dict) -> dict:
    """Parse a Chat Protocol ``Resource`` (``{"uri": ..., "metadata": {...}}``).

    Accepts inline base64 under ``metadata.data``/``metadata.content``, a
    ``data:`` URI, or plain text under ``metadata.text``. Remote URIs are NOT
    fetched: untrusted locations are never retrieved. Returns the same shape
    as :func:`parse_resume` plus ``status``/``filename``.
    """
    uri = str((resource or {}).get("uri") or "")
    metadata = (resource or {}).get("metadata") or {}
    filename = str(metadata.get("filename") or metadata.get("name") or uri.rsplit("/", 1)[-1] or "resume")
    data = None
    for key in ("data", "content", "base64"):
        raw = metadata.get(key)
        if isinstance(raw, str) and raw.strip():
            try:
                data = base64.b64decode(raw, validate=True)
            except (binascii.Error, ValueError):
                data = raw.encode("utf-8")
            break
    if data is None and uri.startswith("data:"):
        _, _, payload = uri.partition(",")
        try:
            data = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError):
            data = payload.encode("utf-8")
    if data is None and isinstance(metadata.get("text"), str):
        data = metadata["text"].encode("utf-8")
    if data is None:
        return {
            "status": "unsupported",
            "filename": filename,
            "error": (
                "the résumé arrived as a link without file contents, and remote files are "
                "never fetched for safety. Paste the résumé text or attach the file itself."
            ),
        }
    parsed = parse_resume_bytes(filename, data)
    parsed["filename"] = filename
    return parsed


def parse_resume_bytes(filename: str, data: bytes, as_of: date | None = None) -> dict:
    """Extract text from bytes and normalize it into a canonical profile fragment."""
    extracted = extract_text(filename, data)
    if extracted["status"] != "ok":
        return extracted
    return parse_resume(extracted["text"], as_of=as_of)


def sanitize(text: str) -> tuple[str, list[str]]:
    """Strip control characters, cap length, and flag instruction-like text."""
    warnings: list[str] = []
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", str(text or ""))
    if len(cleaned) > pf.MAX_TEXT_CHARS:
        warnings.append(f"Résumé truncated to {pf.MAX_TEXT_CHARS} characters before parsing.")
        cleaned = cleaned[:pf.MAX_TEXT_CHARS]
    lowered = cleaned.lower()
    if any(pattern in lowered for pattern in pf.INJECTION_PATTERNS):
        warnings.append(
            "Résumé contained instruction-like text; it was ignored as career data and never acted on."
        )
    return cleaned, warnings


def parse_resume(text: str, as_of: date | None = None) -> dict:
    """Normalize résumé text into ``{"status": "ok", "profile": ..., "warnings": [...]}``.

    Extraction, section detection, fact extraction and normalization share the
    profile module's parser; provenance is always ``resume``.
    """
    cleaned, warnings = sanitize(text)
    try:
        result = pf.normalize_input(cleaned, provenance=pf.PROVENANCE_RESUME, as_of=as_of)
    except ValueError as exc:
        return {"status": "error", "error": f"could not read career facts from the résumé ({exc})."}
    profile = result["profile"]
    profile["source_metadata"].append({"source": pf.PROVENANCE_RESUME, "kind": "resume_text"})
    for warning in result["warnings"]:
        if warning not in warnings:
            warnings.append(warning)
    return {"status": "ok", "profile": profile, "warnings": warnings}
