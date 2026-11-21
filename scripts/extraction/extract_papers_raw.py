from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(r"d:\MY PROJECTS\THESIS")
PAPERS_DIR = ROOT / "thesis_content" / "papers"
OUT_DIR = ROOT / "thesis_content" / "extraction"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "raw_extracted_content.md"

KEYWORDS = [
    "definition",
    "defined",
    "method",
    "methodology",
    "proposed",
    "approach",
    "dataset",
    "data",
    "results",
    "finding",
    "conclusion",
    "experiment",
    "accuracy",
    "performance",
    "fault",
    "diagnosis",
]

SECTION_HEADING_KEYWORDS = [
    "abstract",
    "introduction",
    "background",
    "related work",
    "method",
    "methodology",
    "materials and methods",
    "experimental setup",
    "results",
    "discussion",
    "conclusion",
]

FIGURE_PATTERNS = [
    re.compile(r"\bFigure\s+\d+[A-Za-z-]*[:.\s].*", re.IGNORECASE),
    re.compile(r"\bFig\.\s*\d+[A-Za-z-]*[:.\s].*", re.IGNORECASE),
    re.compile(r"\bTable\s+\d+[A-Za-z-]*[:.\s].*", re.IGNORECASE),
    re.compile(r"\bEq\.\s*\(?\d+\)?[:.\s].*", re.IGNORECASE),
    re.compile(r"\bEquation\s*\(?\d+\)?[:.\s].*", re.IGNORECASE),
    re.compile(r"\bAlgorithm\s+\d+[A-Za-z-]*[:.\s].*", re.IGNORECASE),
]


def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        pages.append(txt)
    return "\n\n".join(pages)


def split_paragraphs(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    chunks = re.split(r"\n\s*\n", text)
    paragraphs = []
    for ch in chunks:
        ch = ch.strip()
        if not ch:
            continue
        # Keep exact wording but normalize excessive internal line breaks.
        ch = re.sub(r"\n{2,}", "\n", ch)
        paragraphs.append(ch)
    return paragraphs


def keyword_match(paragraph: str) -> bool:
    p = paragraph.lower()
    return any(k in p for k in KEYWORDS)


def heading_match(paragraph: str) -> bool:
    p = paragraph.strip().lower()
    p_compact = re.sub(r"\s+", " ", p)
    if len(p_compact) > 90:
        return False
    return any(h in p_compact for h in SECTION_HEADING_KEYWORDS)


def select_important_paragraphs(paragraphs: list[str], max_items: int = 140) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()

    def add_paragraph(p: str) -> None:
        key = p.strip()
        if not key or key in seen:
            return
        seen.add(key)
        selected.append(p)

    # 1) Strong keyword-based extraction (broad).
    for p in paragraphs:
        if keyword_match(p):
            add_paragraph(p)
        if len(selected) >= 90:
            break

    # 2) Section heading blocks: heading + next 4 paragraphs.
    for idx, p in enumerate(paragraphs):
        if heading_match(p):
            for j in range(idx, min(idx + 5, len(paragraphs))):
                add_paragraph(paragraphs[j])
        if len(selected) >= 120:
            break

    # 3) Add long explanatory paragraphs.
    for p in paragraphs:
        if len(p) >= 350:
            add_paragraph(p)
        if len(selected) >= max_items:
            break

    # 4) Fallback: top-of-paper paragraphs.
    if not selected:
        for p in paragraphs[:max_items]:
            add_paragraph(p)

    return selected[:max_items]


def extract_figure_lines(text: str, max_items: int = 120) -> list[str]:
    lines = [ln.strip() for ln in text.replace("\r\n", "\n").split("\n") if ln.strip()]
    out = []
    for idx, ln in enumerate(lines):
        if any(pat.search(ln) for pat in FIGURE_PATTERNS):
            out.append(ln)
            # Keep immediate continuation line if present.
            if idx + 1 < len(lines):
                nxt = lines[idx + 1]
                if not any(pat.search(nxt) for pat in FIGURE_PATTERNS):
                    out.append(nxt)
        if len(out) >= max_items:
            break
    return out[:max_items]


def main() -> None:
    pdfs = sorted(PAPERS_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDF files found in {PAPERS_DIR}")

    lines: list[str] = []
    lines.append("# Raw Extracted Content (Verbatim)")
    lines.append("")
    lines.append("Source folder: thesis_content/papers")
    lines.append("")

    for idx, pdf in enumerate(pdfs, start=1):
        lines.append(f"## Paper {idx}: {pdf.name}")
        lines.append("")
        try:
            text = extract_text_from_pdf(pdf)
            paragraphs = split_paragraphs(text)
            important = select_important_paragraphs(paragraphs)
            figures = extract_figure_lines(text)

            lines.append("### Extracted Important Paragraphs (Exact, Broader Pull)")
            lines.append("")
            if important:
                for p in important:
                    lines.append("```text")
                    lines.append(p)
                    lines.append("```")
                    lines.append("")
            else:
                lines.append("(No extractable text found.)")
                lines.append("")

            lines.append("### Extracted Figure/Table/Equation Lines (Exact)")
            lines.append("")
            if figures:
                for f in figures:
                    lines.append("```text")
                    lines.append(f)
                    lines.append("```")
                    lines.append("")
            else:
                lines.append("(No figure-caption text detected in extracted PDF text.)")
                lines.append("")

        except Exception as exc:  # noqa: BLE001
            lines.append("### Extraction Error")
            lines.append("")
            lines.append("```text")
            lines.append(f"{type(exc).__name__}: {exc}")
            lines.append("```")
            lines.append("")

    OUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(str(OUT_FILE))


if __name__ == "__main__":
    main()
