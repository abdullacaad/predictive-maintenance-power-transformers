from __future__ import annotations

import re
from datetime import date
from datetime import datetime
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, Inches

ROOT = Path(r"d:\MY PROJECTS\THESIS data science")
INPUT_MD = ROOT / "report" / "thesis_content" / "full_msc_thesis_code_aligned.md"
OUTPUT_DOCX = ROOT / "MSc_Thesis_Ready_Final.docx"

STUDENT_NUMBER = None
SUBMISSION_DATE = date(2026, 3, 26).strftime("%d %B %Y")


def normalize_inline(text: str) -> str:
    text = text.replace("`", "")
    text = text.replace("**", "")
    text = text.replace("*", "")
    return text.strip()


def configure_document_styles(doc: Document) -> None:
    section = doc.sections[0]
    section.top_margin = Inches(0.5)
    section.bottom_margin = Inches(0.5)
    section.left_margin = Inches(0.5)
    section.right_margin = Inches(0.5)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    normal.font.size = Pt(9)

    pf = normal.paragraph_format
    pf.line_spacing = 1.5
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    for style_name in ["Heading 1", "Heading 2", "Heading 3"]:
        style = doc.styles[style_name]
        style.font.name = "Times New Roman"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        style.font.size = Pt(9)
        style.font.bold = True


def add_paragraph(doc: Document, text: str) -> None:
    p = doc.add_paragraph(normalize_inline(text))
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


def add_table(doc: Document, rows: list[str]) -> None:
    parsed = []
    for r in rows:
        if not r.strip() or re.match(r"^\|\s*-+", r):
            continue
        cells = [normalize_inline(c) for c in r.strip().strip("|").split("|")]
        parsed.append(cells)

    if not parsed:
        return

    max_cols = max(len(r) for r in parsed)
    table = doc.add_table(rows=1, cols=max_cols)
    table.style = "Table Grid"

    for i, cell_text in enumerate(parsed[0]):
        table.rows[0].cells[i].text = cell_text

    for data_row in parsed[1:]:
        row_cells = table.add_row().cells
        for i, cell_text in enumerate(data_row):
            row_cells[i].text = cell_text


def build_doc() -> None:
    raw = INPUT_MD.read_text(encoding="utf-8")
    if STUDENT_NUMBER:
        raw = raw.replace("[Student Number]", STUDENT_NUMBER)
    raw = raw.replace("[Date]", SUBMISSION_DATE)

    lines = raw.splitlines()

    doc = Document()
    configure_document_styles(doc)

    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\n")
        stripped = line.strip()

        if not stripped:
            doc.add_paragraph("")
            i += 1
            continue

        if stripped == "---":
            doc.add_page_break()
            i += 1
            continue

        if stripped.startswith("### "):
            doc.add_heading(normalize_inline(stripped[4:]), level=3)
            i += 1
            continue

        if stripped.startswith("## "):
            doc.add_heading(normalize_inline(stripped[3:]), level=2)
            i += 1
            continue

        if stripped.startswith("# "):
            doc.add_heading(normalize_inline(stripped[2:]), level=1)
            i += 1
            continue

        if stripped.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            add_table(doc, table_lines)
            continue

        m = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if m:
            p = doc.add_paragraph(style="List Number")
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.add_run(normalize_inline(m.group(2)))
            i += 1
            continue

        add_paragraph(doc, stripped)
        i += 1

    doc.save(OUTPUT_DOCX)


if __name__ == "__main__":
    try:
        build_doc()
        print(f"Created: {OUTPUT_DOCX}")
    except PermissionError:
        fallback = OUTPUT_DOCX.with_name(
            f"{OUTPUT_DOCX.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{OUTPUT_DOCX.suffix}"
        )
        # Rebuild document and save to fallback output when the main file is locked.
        OUTPUT_DOCX = fallback
        build_doc()
        print(f"Created fallback: {OUTPUT_DOCX}")
