"""
Replace embedded images in manuscript.docx and supplementary_figures.docx
with the current PNG files, leaving all text and formatting untouched.

Run this whenever figures are regenerated (e.g. after running generate_figures.py).
"""
import os
from docx import Document

DIR   = os.path.dirname(os.path.abspath(__file__))
WP_NS = 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'
A_NS  = 'http://schemas.openxmlformats.org/drawingml/2006/main'
R_NS  = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'


def refresh_images(docx_path):
    doc = Document(docx_path)
    updated = 0

    for para in doc.paragraphs:
        blips = para._element.findall(f'.//{{{A_NS}}}blip')
        if not blips:
            continue

        docPr = para._element.find(f'.//{{{WP_NS}}}docPr')
        if docPr is None:
            continue

        filename = docPr.get('descr', '')
        if not filename:
            continue

        img_path = os.path.join(DIR, filename)
        if not os.path.exists(img_path):
            print(f"  WARNING: {filename} not found, skipping")
            continue

        with open(img_path, 'rb') as f:
            new_blob = f.read()

        for blip in blips:
            rId = blip.get(f'{{{R_NS}}}embed')
            if rId and rId in doc.part.related_parts:
                doc.part.related_parts[rId]._blob = new_blob
                updated += 1

    doc.save(docx_path)
    print(f"  {updated} image(s) refreshed → {os.path.basename(docx_path)}")


for docx_file in ["manuscript.docx", "supplementary_figures.docx"]:
    path = os.path.join(DIR, docx_file)
    if os.path.exists(path):
        print(f"Refreshing {docx_file}...")
        refresh_images(path)
    else:
        print(f"Skipping {docx_file} (not found)")
