"""Generate PDF versions of PRESENTATION_GUIDE.md and PROJECT_REPORT.md using fpdf2."""
from __future__ import annotations

import re
from pathlib import Path

from fpdf import FPDF


class MarkdownPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)
        self.add_page()
        self.set_left_margin(15)
        self.set_right_margin(15)

    def _write_heading(self, level: int, text: str):
        sizes = {1: 22, 2: 16, 3: 13, 4: 11}
        size = sizes.get(level, 11)
        self.ln(4 if level <= 2 else 2)
        self.set_font("Helvetica", "B", size)
        self.multi_cell(0, size * 0.55, text)
        if level == 1:
            self.set_draw_color(41, 128, 185)
            self.set_line_width(0.8)
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
            self.ln(3)
        elif level == 2:
            self.set_draw_color(180, 180, 180)
            self.set_line_width(0.3)
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
            self.ln(2)
        else:
            self.ln(1)

    def _write_paragraph(self, text: str):
        self.set_font("Helvetica", "", 10)
        clean = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        clean = re.sub(r'`(.+?)`', r'\1', clean)
        self.multi_cell(0, 5, clean)
        self.ln(1)

    def _write_bold_paragraph(self, text: str):
        self.set_font("Helvetica", "B", 10)
        clean = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        clean = re.sub(r'`(.+?)`', r'\1', clean)
        self.multi_cell(0, 5, clean)
        self.ln(1)

    def _write_bullet(self, text: str):
        self.set_font("Helvetica", "", 10)
        clean = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        clean = re.sub(r'`(.+?)`', r'\1', clean)
        x = self.get_x()
        self.cell(8, 5, "-")
        self.multi_cell(0, 5, clean.strip())
        self.ln(0.5)

    def _write_numbered(self, num: str, text: str):
        self.set_font("Helvetica", "", 10)
        clean = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        clean = re.sub(r'`(.+?)`', r'\1', clean)
        self.cell(8, 5, f"{num}.")
        self.multi_cell(0, 5, clean.strip())
        self.ln(0.5)

    def _write_code_block(self, lines: list[str]):
        self.set_font("Courier", "", 8)
        self.set_fill_color(240, 240, 240)
        for line in lines:
            safe = line.replace("\t", "    ")
            safe = safe.encode('latin-1', errors='replace').decode('latin-1')
            self.cell(0, 4, f"  {safe}", new_x="LMARGIN", new_y="NEXT", fill=True)
        self.ln(2)

    def _write_table(self, rows: list[list[str]]):
        if not rows or len(rows) < 2:
            return
        headers = rows[0]
        data = [r for r in rows[1:] if not all(c.strip() == '' or set(c.strip()) <= {'-', '|'} for c in r)]
        n_cols = len(headers)
        available = self.w - self.l_margin - self.r_margin
        col_w = available / max(1, n_cols)

        self.set_font("Helvetica", "B", 8)
        self.set_fill_color(41, 128, 185)
        self.set_text_color(255, 255, 255)
        for h in headers:
            clean = h.strip().replace('**', '')
            self.cell(col_w, 6, clean[:25], border=1, fill=True)
        self.ln()
        self.set_text_color(0, 0, 0)

        self.set_font("Helvetica", "", 8)
        fill = False
        for row in data:
            if fill:
                self.set_fill_color(245, 245, 245)
            else:
                self.set_fill_color(255, 255, 255)
            for i, cell in enumerate(row):
                clean = cell.strip().replace('**', '').replace('`', '')
                self.cell(col_w, 5, clean[:30], border=1, fill=True)
            self.ln()
            fill = not fill
        self.ln(2)

    def _write_divider(self):
        self.ln(2)
        self.set_draw_color(200, 200, 200)
        self.set_line_width(0.2)
        y = self.get_y()
        self.line(self.l_margin, y, self.w - self.r_margin, y)
        self.ln(3)

    def render_markdown(self, md_text: str):
        lines = md_text.split('\n')
        i = 0
        in_code = False
        code_lines: list[str] = []
        table_rows: list[list[str]] = []

        while i < len(lines):
            line = lines[i]

            # Code blocks
            if line.strip().startswith('```'):
                if in_code:
                    self._write_code_block(code_lines)
                    code_lines = []
                    in_code = False
                else:
                    if table_rows:
                        self._write_table(table_rows)
                        table_rows = []
                    in_code = True
                i += 1
                continue

            if in_code:
                code_lines.append(line)
                i += 1
                continue

            # Table rows
            if '|' in line and line.strip().startswith('|'):
                cells = [c.strip() for c in line.split('|')]
                cells = [c for c in cells if c != '']
                if all(set(c.strip()) <= {'-', ':', ' '} for c in cells):
                    i += 1
                    continue
                table_rows.append(cells)
                i += 1
                continue
            else:
                if table_rows:
                    self._write_table(table_rows)
                    table_rows = []

            stripped = line.strip()

            # Empty line
            if not stripped:
                i += 1
                continue

            # Horizontal rule
            if stripped in ('---', '***', '___'):
                self._write_divider()
                i += 1
                continue

            # Headings
            heading_match = re.match(r'^(#{1,4})\s+(.+)$', stripped)
            if heading_match:
                level = len(heading_match.group(1))
                text = heading_match.group(2)
                self._write_heading(level, text)
                i += 1
                continue

            # Numbered list
            num_match = re.match(r'^(\d+)\.\s+(.+)$', stripped)
            if num_match:
                self._write_numbered(num_match.group(1), num_match.group(2))
                i += 1
                continue

            # Bullet
            if stripped.startswith('- ') or stripped.startswith('* '):
                self._write_bullet(stripped[2:])
                i += 1
                continue

            # Bold line
            if stripped.startswith('**') and stripped.endswith('**'):
                self._write_bold_paragraph(stripped)
                i += 1
                continue

            # Regular paragraph
            self._write_paragraph(stripped)
            i += 1

        if table_rows:
            self._write_table(table_rows)
        if code_lines:
            self._write_code_block(code_lines)


def _sanitize(text: str) -> str:
    replacements = {
        '\u2014': '-', '\u2013': '-', '\u2018': "'", '\u2019': "'",
        '\u201c': '"', '\u201d': '"', '\u2026': '...', '\u2192': '->',
        '\u2190': '<-', '\u2191': '^', '\u2193': 'v', '\u2022': '-',
        '\u25cf': '-', '\u25cb': 'o', '\u2713': '[x]', '\u2717': '[ ]',
        '\u2716': 'x', '\u00d7': 'x', '\u2265': '>=', '\u2264': '<=',
        '\u2260': '!=', '\u221e': 'inf',
        '\u250c': '+', '\u2510': '+', '\u2514': '+', '\u2518': '+',
        '\u2500': '-', '\u2502': '|', '\u251c': '+', '\u2524': '+',
        '\u252c': '+', '\u2534': '+', '\u253c': '+',
        '\u2550': '=', '\u2551': '|', '\u2552': '+', '\u2553': '+',
        '\u2554': '+', '\u2555': '+', '\u2556': '+', '\u2557': '+',
        '\u2558': '+', '\u2559': '+', '\u255a': '+', '\u255b': '+',
        '\u255c': '+', '\u255d': '+', '\u255e': '+', '\u255f': '+',
        '\u2560': '+', '\u2561': '+', '\u2562': '+', '\u2563': '+',
        '\u2564': '+', '\u2565': '+', '\u2566': '+', '\u2567': '+',
        '\u2568': '+', '\u2569': '+', '\u256a': '+', '\u256b': '+',
        '\u256c': '+',
        '\u25b6': '>', '\u25ba': '>', '\u25b8': '>',
        '\u25c0': '<', '\u25c4': '<',
        '\u25bc': 'v', '\u25be': 'v',
        '\u25b2': '^', '\u25b4': '^',
        '\u2588': '#', '\u2591': '.', '\u2592': ':', '\u2593': '#',
        '\u2500': '-', '\u2501': '=',
        '\u2577': '|', '\u2575': '|', '\u2576': '-', '\u2574': '-',
        '\u256d': '+', '\u256e': '+', '\u256f': '+', '\u2570': '+',
        '\u2571': '/', '\u2572': '\\',
        '\u2581': '_', '\u2582': '_', '\u2583': '_',
        '\u2584': '_', '\u2585': '_', '\u2586': '_', '\u2587': '_',
        '\u25ac': '=',
        '\u2610': '[ ]', '\u2611': '[x]', '\u2612': '[x]',
        '\u2796': '-', '\u2795': '+',
        '\u279c': '->', '\u279e': '->',
        '\u25aa': '-', '\u25ab': '-', '\u25a0': '#', '\u25a1': '[ ]',
        '\u2714': '[x]', '\u2718': '[ ]',
        '\u21b3': '->', '\u21d2': '=>',
        '\u258c': '|', '\u2590': '|',
        '\u2594': '-', '\u2015': '-',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Catch any remaining non-latin-1 characters
    text = text.encode('latin-1', errors='replace').decode('latin-1')
    return text


def convert_md_to_pdf(md_path: str, pdf_path: str):
    text = Path(md_path).read_text(encoding='utf-8')
    text = _sanitize(text)
    pdf = MarkdownPDF()
    pdf.render_markdown(text)
    pdf.output(pdf_path)
    print(f"Generated: {pdf_path}")


if __name__ == "__main__":
    docs = Path(__file__).parent
    convert_md_to_pdf(
        str(docs / "PRESENTATION_GUIDE.md"),
        str(docs / "PRESENTATION_GUIDE.pdf"),
    )
    convert_md_to_pdf(
        str(docs / "PROJECT_REPORT.md"),
        str(docs / "PROJECT_REPORT.pdf"),
    )
