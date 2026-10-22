"""Export the existing EDA Markdown and figures as a three-page PDF."""
from pathlib import Path
import re
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'from_scratch/runs/eda/oxford_iiit_pets'
OUTPUT = ROOT / 'output/pdf/TinySAM_EDA_Report.pdf'
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyEDA', fontName='Helvetica', fontSize=9, leading=12, spaceAfter=7))
styles.add(ParagraphStyle(name='CaptionEDA', fontSize=8, leading=10, textColor=colors.HexColor('#526174'), alignment=TA_CENTER, spaceAfter=8))
styles['Heading1'].fontSize = 17
styles['Heading1'].leading = 21
styles['Heading2'].fontSize = 12
styles['Heading2'].leading = 15

def paragraph(text):
    text = escape(text)
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'`(.*?)`', r'<font name="Courier">\1</font>', text)
    return Paragraph(text, styles['BodyEDA'])

def figure(name, max_height):
    with PILImage.open(SOURCE / name) as picture:
        width, height = picture.size
    scale = min(475 / width, max_height / height)
    return Image(str(SOURCE / name), width=width * scale, height=height * scale)

def footer(canvas, doc):
    canvas.setStrokeColor(colors.HexColor('#d8e0e8'))
    canvas.line(42, 36, A4[0]-42, 36)
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(colors.HexColor('#526174'))
    canvas.drawString(42, 24, 'TinySAM | Oxford-IIIT Pets | Exploratory Data Analysis')
    canvas.drawRightString(A4[0]-42, 24, str(doc.page))

text = (SOURCE / 'eda_report.md').read_text(encoding='utf-8')
sections = re.split(r'^## ', text, flags=re.MULTILINE)
story = [Paragraph('TinySAM: Exploratory Data Analysis', styles['Heading1']),
         paragraph('Oxford-IIIT Pets | Raw dataset audit | Python 3.14'), Spacer(1, 5)]

for index, section in enumerate(sections[1:], start=1):
    heading, body = section.split('\n', 1)
    if index in (2, 3):
        story.append(PageBreak())
    story.append(Paragraph(escape(heading), styles['Heading2']))
    for block in re.split(r'\n\s*\n', body.strip()):
        if block.startswith('|'):
            rows = [[cell.strip() for cell in line.strip().strip('|').split('|')]
                    for line in block.splitlines() if not re.match(r'^\|[-:| ]+\|$', line)]
            table = Table(rows, colWidths=[230, 230])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#173b57')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,-1), 9),
                ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                ('TOPPADDING', (0,0), (-1,-1), 6),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#edf2f6'), colors.white]),
            ]))
            story.extend([table, Spacer(1, 9)])
        elif block.startswith('!['):
            caption, name = re.match(r'!\[(.*?)\]\((.*?)\)', block).groups()
            # Put the compact coverage plot on page 3 alongside image geometry.
            if name == 'mask_coverage.png':
                continue
            limit = {'breed_distribution.png': 325, 'raw_examples.png': 465, 'image_geometry.png': 130}[name]
            story.extend([figure(name, limit), Paragraph(escape(caption), styles['CaptionEDA'])])
            if name == 'image_geometry.png':
                story.extend([figure('mask_coverage.png', 130), Paragraph('Project foreground and not-classified-label coverage', styles['CaptionEDA'])])
        else:
            story.append(paragraph(' '.join(block.splitlines())))

document = SimpleDocTemplate(str(OUTPUT), pagesize=A4, rightMargin=42, leftMargin=42,
                            topMargin=36, bottomMargin=48, title='TinySAM EDA Report', author='TinySAM Project')
document.build(story, onFirstPage=footer, onLaterPages=footer)
print(OUTPUT)
