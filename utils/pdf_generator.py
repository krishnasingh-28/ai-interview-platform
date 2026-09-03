"""PDF Report Generator for AI Interview Integrity sessions using ReportLab."""
from __future__ import annotations

import html
import io
import re
from datetime import datetime
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from monitoring.models import IntegrityEvent, SessionMetrics
from utils.time_utils import format_seconds

# Palette constants
PRIMARY_NAVY = colors.HexColor("#0F172A")
SECONDARY_BLUE = colors.HexColor("#2563EB")
ACCENT_CYAN = colors.HexColor("#0EA5E9")
TEXT_MAIN = colors.HexColor("#1E293B")
TEXT_MUTED = colors.HexColor("#64748B")
BG_LIGHT = colors.HexColor("#F8FAFC")
BG_CARD = colors.HexColor("#F1F5F9")
BORDER_COLOR = colors.HexColor("#CBD5E1")
ALERT_RED = colors.HexColor("#EF4444")
ALERT_AMBER = colors.HexColor("#F59E0B")
ALERT_GREEN = colors.HexColor("#10B981")

EMOJI_REPLACEMENTS = {
    "🎯": "[Overview]",
    "👁️": "[Attention]",
    "👁": "[Attention]",
    "📱": "[Device]",
    "💡": "[Guidance]",
    "🛡️": "[Ethics]",
    "🛡": "[Ethics]",
    "🔴": "[ALERT]",
    "🟡": "[CAUTION]",
    "✅": "[PASS]",
    "⚠️": "[WARNING]",
    "ℹ️": "[INFO]",
    "✨": "[AI]",
    "📋": "[REPORT]",
    "✓": "[OK]",
    "📥": "[DOWNLOAD]",
    "🤖": "[AI]",
}


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and draw total page count and running header/footer."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._saved_page_states: list[dict[str, Any]] = []

    def showPage(self) -> None:
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count: int) -> None:
        self.saveState()
        
        # Header (pages 2+)
        if self._pageNumber > 1:
            self.setFont("Helvetica", 8)
            self.setFillColor(TEXT_MUTED)
            self.drawString(36, 756, "AI Interview Integrity Analytical Summary")
            self.drawRightString(612 - 36, 756, "Confidential Candidate Report")
            self.setStrokeColor(BORDER_COLOR)
            self.setLineWidth(0.5)
            self.line(36, 750, 612 - 36, 750)

        # Running Footer (all pages)
        self.setStrokeColor(BORDER_COLOR)
        self.setLineWidth(0.5)
        self.line(36, 38, 612 - 36, 38)

        self.setFont("Helvetica", 8)
        self.setFillColor(TEXT_MUTED)
        self.drawString(36, 26, "AI Interview Integrity Platform • Observational Assessment")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(612 - 36, 26, page_str)

        self.restoreState()


def _sanitize_markdown_text(raw_text: str) -> str:
    """Replaces emojis and unsupported Unicode glyphs with safe ASCII representations."""
    text = raw_text
    for emoji, rep in EMOJI_REPLACEMENTS.items():
        text = text.replace(emoji, rep)

    # Replace unsupported high Unicode characters with ascii or space
    cleaned: list[str] = []
    for ch in text:
        code = ord(ch)
        if code > 127 and code not in (0x2013, 0x2014, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x00A0):
            cleaned.append(" ")
        else:
            cleaned.append(ch)
    return "".join(cleaned)


def _format_inline_markdown(raw_line: str) -> str:
    """Converts bold, italics, inline code, and XML escaping for ReportLab Paragraph markup."""
    line = _sanitize_markdown_text(raw_line)

    # First escape XML special characters
    line = html.escape(line, quote=False)

    # Bold: **text** or __text__ -> <b>text</b>
    line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)
    line = re.sub(r"__(.+?)__", r"<b>\1</b>", line)

    # Italics: *text* or _text_ -> <i>text</i>
    line = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", line)
    line = re.sub(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)", r"<i>\1</i>", line)

    # Inline code: `code` -> <font face="Courier">code</font>
    line = re.sub(r"`(.+?)`", r'<font face="Courier" color="#1E293B"><b>\1</b></font>', line)

    return line


def parse_markdown_to_flowables(markdown_text: str, styles: dict[str, ParagraphStyle]) -> list[Any]:
    """Parses a markdown string into a list of styled ReportLab Flowables."""
    flowables: list[Any] = []
    lines = markdown_text.strip().split("\n")
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if not line:
            i += 1
            continue

        # Headings
        if line.startswith("### "):
            header_text = _format_inline_markdown(line[4:].strip())
            flowables.append(Spacer(1, 8))
            flowables.append(Paragraph(header_text, styles["SectionHeading"]))
            flowables.append(Spacer(1, 4))
            i += 1
            continue

        if line.startswith("## "):
            header_text = _format_inline_markdown(line[3:].strip())
            flowables.append(Spacer(1, 10))
            flowables.append(Paragraph(header_text, styles["MajorHeading"]))
            flowables.append(Spacer(1, 5))
            i += 1
            continue

        if line.startswith("# "):
            header_text = _format_inline_markdown(line[2:].strip())
            flowables.append(Spacer(1, 12))
            flowables.append(Paragraph(header_text, styles["DocTitle"]))
            flowables.append(Spacer(1, 6))
            i += 1
            continue

        # Blockquote / Callout
        if line.startswith("> "):
            quote_lines = []
            while i < len(lines) and lines[i].strip().startswith("> "):
                quote_lines.append(lines[i].strip()[2:].strip())
                i += 1
            quote_text = _format_inline_markdown(" ".join(quote_lines))
            quote_p = Paragraph(quote_text, styles["CalloutText"])
            quote_table = Table([[quote_p]], colWidths=[540])
            quote_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), BG_CARD),
                ("BOX", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
                ("LINELEFT", (0, 0), (0, 0), 3.0, SECONDARY_BLUE),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ]))
            flowables.append(quote_table)
            flowables.append(Spacer(1, 6))
            continue

        # Bullet items
        if line.startswith(("- ", "* ", "• ")):
            bullet_text = _format_inline_markdown(line[2:].strip())
            flowables.append(Paragraph(f"&bull; {bullet_text}", styles["BulletItem"]))
            i += 1
            continue

        # Numbered list
        num_match = re.match(r"^(\d+)\.\s+(.*)$", line)
        if num_match:
            num_str, item_text = num_match.groups()
            formatted_text = _format_inline_markdown(item_text)
            flowables.append(Paragraph(f"<b>{num_str}.</b> {formatted_text}", styles["BulletItem"]))
            i += 1
            continue

        # Standard Paragraph
        p_text = _format_inline_markdown(line)
        flowables.append(Paragraph(p_text, styles["Body"]))
        flowables.append(Spacer(1, 4))
        i += 1

    return flowables


def build_metrics_table(metrics: SessionMetrics, styles: dict[str, ParagraphStyle]) -> Table:
    """Builds a structured 4-column key performance indicator table."""
    elapsed = max(metrics.elapsed_seconds, 0.1)
    face_absent_pct = min(100.0, (metrics.face_absent_seconds / elapsed) * 100.0)
    face_present_pct = max(0.0, 100.0 - face_absent_pct)
    gaze_away_pct = min(100.0, (metrics.gaze_deviation_seconds / elapsed) * 100.0)
    gaze_center_pct = max(0.0, 100.0 - gaze_away_pct)
    head_turn_pct = min(100.0, (metrics.head_turn_seconds / elapsed) * 100.0)
    head_stable_pct = max(0.0, 100.0 - head_turn_pct)

    data = [
        [
            Paragraph("<b>Session Duration</b>", styles["TableLabel"]),
            Paragraph(f"<b>{format_seconds(metrics.elapsed_seconds)}</b> ({metrics.elapsed_seconds:.1f}s)", styles["TableValue"]),
            Paragraph("<b>Frames Processed</b>", styles["TableLabel"]),
            Paragraph(f"<b>{metrics.frame_count}</b> ({metrics.average_fps:.1f} FPS)", styles["TableValue"]),
        ],
        [
            Paragraph("<b>Face Presence</b>", styles["TableLabel"]),
            Paragraph(f"<b>{face_present_pct:.1f}%</b> ({metrics.face_absent_seconds:.1f}s absent)", styles["TableValue"]),
            Paragraph("<b>Gaze Centered</b>", styles["TableLabel"]),
            Paragraph(f"<b>{gaze_center_pct:.1f}%</b> ({metrics.gaze_deviation_seconds:.1f}s away)", styles["TableValue"]),
        ],
        [
            Paragraph("<b>Head Stability</b>", styles["TableLabel"]),
            Paragraph(f"<b>{head_stable_pct:.1f}%</b> ({metrics.head_turn_seconds:.1f}s turn)", styles["TableValue"]),
            Paragraph("<b>Secondary Devices</b>", styles["TableLabel"]),
            Paragraph(f"<b>{metrics.phone_detection_count}</b> episode(s)", styles["TableValueAlert" if metrics.phone_detection_count > 0 else "TableValue"]),
        ],
        [
            Paragraph("<b>Multiple Persons</b>", styles["TableLabel"]),
            Paragraph(f"<b>{metrics.multiple_face_count}</b> episode(s)", styles["TableValueAlert" if metrics.multiple_face_count > 0 else "TableValue"]),
            Paragraph("<b>Avg Frame Rate</b>", styles["TableLabel"]),
            Paragraph(f"<b>{metrics.average_fps:.1f} FPS</b>", styles["TableValue"]),
        ],
    ]

    table = Table(data, colWidths=[130, 140, 130, 140])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BG_LIGHT),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return table


def build_events_table(events: list[IntegrityEvent], styles: dict[str, ParagraphStyle]) -> Table:
    """Builds a formatted log table of flagged integrity incidents."""
    headers = [
        Paragraph("<b>#</b>", styles["TableHeader"]),
        Paragraph("<b>Time (s)</b>", styles["TableHeader"]),
        Paragraph("<b>Event Type / Flag</b>", styles["TableHeader"]),
        Paragraph("<b>Duration</b>", styles["TableHeader"]),
        Paragraph("<b>Confidence</b>", styles["TableHeader"]),
    ]

    rows: list[list[Any]] = [headers]
    for idx, ev in enumerate(events[:25], start=1):  # Cap at 25 recent events to prevent overflow
        duration_val = max(ev.duration, (ev.end_time - ev.start_time) if ev.end_time else ev.duration)
        conf_str = f"{ev.confidence:.2f}" if ev.confidence is not None else "N/A"
        
        is_severe = ev.event_type in {"FACE_NOT_VISIBLE", "MULTIPLE_FACES", "POSSIBLE_SECONDARY_DEVICE"}
        ev_style = styles["TableValueAlert"] if is_severe else styles["TableValue"]

        rows.append([
            Paragraph(str(idx), styles["TableValue"]),
            Paragraph(f"{ev.start_time:.1f}s", styles["TableValue"]),
            Paragraph(f"<b>{ev.event_type.replace('_', ' ')}</b>", ev_style),
            Paragraph(f"{duration_val:.1f}s", styles["TableValue"]),
            Paragraph(conf_str, styles["TableValue"]),
        ])

    table = Table(rows, colWidths=[30, 80, 250, 90, 90])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY_NAVY),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return table


def generate_pdf_report(
    summary_markdown: str,
    metrics: SessionMetrics | None = None,
    events: list[IntegrityEvent] | None = None,
    model_used: str = "AI Layman Analytical Engine",
) -> bytes:
    """Generates a complete, professionally styled PDF document of the interview integrity summary.

    Args:
        summary_markdown: The markdown content of the layman/AI analytical summary.
        metrics: Optional SessionMetrics instance containing quantitative session metrics.
        events: Optional list of IntegrityEvent objects recorded during the interview.
        model_used: Identifier of the AI model or analytical engine used.

    Returns:
        bytes: Raw binary PDF data suitable for streaming or download.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=40,
        bottomMargin=48,
    )

    base_styles = getSampleStyleSheet()

    custom_styles: dict[str, ParagraphStyle] = {
        "DocTitle": ParagraphStyle(
            "DocTitle",
            parent=base_styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=PRIMARY_NAVY,
            spaceAfter=2,
        ),
        "Subtitle": ParagraphStyle(
            "Subtitle",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=TEXT_MUTED,
        ),
        "BadgeText": ParagraphStyle(
            "BadgeText",
            parent=base_styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=SECONDARY_BLUE,
        ),
        "MajorHeading": ParagraphStyle(
            "MajorHeading",
            parent=base_styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=PRIMARY_NAVY,
            spaceBefore=8,
            spaceAfter=4,
        ),
        "SectionHeading": ParagraphStyle(
            "SectionHeading",
            parent=base_styles["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=SECONDARY_BLUE,
            spaceBefore=6,
            spaceAfter=2,
        ),
        "Body": ParagraphStyle(
            "Body",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=TEXT_MAIN,
        ),
        "BulletItem": ParagraphStyle(
            "BulletItem",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            leftIndent=12,
            firstLineIndent=-8,
            textColor=TEXT_MAIN,
            spaceAfter=2,
        ),
        "CalloutText": ParagraphStyle(
            "CalloutText",
            parent=base_styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8.5,
            leading=12,
            textColor=TEXT_MAIN,
        ),
        "TableLabel": ParagraphStyle(
            "TableLabel",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=TEXT_MUTED,
        ),
        "TableValue": ParagraphStyle(
            "TableValue",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=TEXT_MAIN,
        ),
        "TableValueAlert": ParagraphStyle(
            "TableValueAlert",
            parent=base_styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=11,
            textColor=ALERT_RED,
        ),
        "TableHeader": ParagraphStyle(
            "TableHeader",
            parent=base_styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=11,
            textColor=colors.white,
        ),
        "DisclaimerText": ParagraphStyle(
            "DisclaimerText",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10.5,
            textColor=TEXT_MUTED,
        ),
    }

    story: list[Any] = []

    # 1. Header Banner
    timestamp_str = datetime.now().strftime("%B %d, %Y - %H:%M:%S")
    header_content = [
        [
            Paragraph("AI Interview Integrity & Layman Analytical Report", custom_styles["DocTitle"]),
            Paragraph(f"<b>Generated:</b> {timestamp_str}<br/><b>Engine:</b> {_sanitize_markdown_text(model_used)}", custom_styles["Subtitle"]),
        ]
    ]
    header_table = Table(header_content, colWidths=[360, 180])
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(header_table)

    # Accent Divider
    story.append(HRFlowable(width="100%", thickness=1.5, color=SECONDARY_BLUE, spaceBefore=4, spaceAfter=8))

    # 2. Executive Metrics Summary Grid (if metrics provided)
    if metrics is not None:
        story.append(Paragraph("<b>Session Performance & Vision Metrics Summary</b>", custom_styles["MajorHeading"]))
        story.append(Spacer(1, 2))
        story.append(build_metrics_table(metrics, custom_styles))
        story.append(Spacer(1, 8))

    # 3. AI Layman Analytical Evaluation
    story.append(Paragraph("<b>AI Layman Analytical Assessment</b>", custom_styles["MajorHeading"]))
    story.append(Spacer(1, 2))
    parsed_flowables = parse_markdown_to_flowables(summary_markdown, custom_styles)
    story.extend(parsed_flowables)
    story.append(Spacer(1, 8))

    # 4. Flagged Integrity Incidents Log Table (if events provided)
    if events is not None:
        incident_elements: list[Any] = [
            Paragraph("<b>Flagged Integrity Events & Incident Log</b>", custom_styles["MajorHeading"]),
            Spacer(1, 2),
        ]
        if len(events) > 0:
            incident_elements.append(build_events_table(events, custom_styles))
        else:
            no_event_p = Paragraph("<b>[OK] Clean Session:</b> No integrity flags or anomalous behaviors were recorded during this interview.", custom_styles["Body"])
            no_event_box = Table([[no_event_p]], colWidths=[540])
            no_event_box.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), BG_CARD),
                ("BOX", (0, 0), (-1, -1), 0.5, ALERT_GREEN),
                ("LINELEFT", (0, 0), (0, 0), 3.0, ALERT_GREEN),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]))
            incident_elements.append(no_event_box)

        story.append(KeepTogether(incident_elements))
        story.append(Spacer(1, 8))

    # 5. Ethical Guidelines & Context Disclaimer
    disclaimer_p = Paragraph(
        "<b>Important Notice & Ethical Considerations:</b> Visual and computer vision tracking metrics are observational signals designed to assist human interviewers. "
        "They do not establish dishonesty, intent, or misconduct. Normal thinking behaviors (e.g., looking away while formulating ideas) or environmental factors must be evaluated by a qualified human reviewer.",
        custom_styles["DisclaimerText"],
    )
    disclaimer_box = Table([[disclaimer_p]], colWidths=[540])
    disclaimer_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BG_CARD),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("LINELEFT", (0, 0), (0, 0), 2.5, TEXT_MUTED),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(KeepTogether([disclaimer_box]))

    # Build document with custom NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    return buffer.getvalue()
