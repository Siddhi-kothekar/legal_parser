"""
Generate a sample case folder under test_data/ for manual upload testing.
This script will:
- Convert the sample text files into PDFs (legal/medical/witness) using reportlab (if available)
- Create placeholder person and scene images with PIL
- Create a structural folder `sample_case_CR_2024_001234/raw_files/` and place the files
- Create `case_metadata.json` with basic case info

Usage:
    python generate_sample_case.py

Requires: reportlab, pillow
"""

from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont
import shutil
import sys

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "sample_case_CR_2024_001234" / "raw_files"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Create PDFs from text files (use create_test_pdfs.text_to_pdf if available)
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    def text_to_pdf(text_path: Path, pdf_path: Path):
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        c = canvas.Canvas(str(pdf_path), pagesize=letter)
        width, height = letter
        c.setFont("Helvetica", 10)
        margin = 72
        y = height - margin
        line_height = 12
        for line in text.split('\n'):
            if y < margin:
                c.showPage()
                y = height - margin
                c.setFont("Helvetica", 10)
            c.drawString(margin, y, line[:120])
            y -= line_height
        c.save()
except Exception:
    def text_to_pdf(text_path: Path, pdf_path: Path):
        # fallback: simply copy text file and rename to .pdf for testing
        shutil.copy(str(text_path), str(pdf_path))

def create_placeholder_image(path: Path, text: str, size=(800,600), bgcolor=(48,54,69), fg=(255,255,255)):
    img = Image.new('RGB', size, color=bgcolor)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
        text_w, text_h = draw.textsize(text, font=font)
        draw.text(((size[0]-text_w)//2, (size[1]-text_h)//2), text, fill=fg, font=font)
    except Exception:
        draw.text((20,20), text, fill=fg)
    img.save(str(path), quality=85)

def copy_and_convert_text_files():
    # Text sources in test_data
    sources = [
        (ROOT / "sample_fir.txt", OUT_DIR / "FIR.pdf"),
        (ROOT / "sample_witness_statement.txt", OUT_DIR / "witness_statement.pdf"),
        (ROOT / "sample_medical_report.txt", OUT_DIR / "medical_report.pdf"),
        (ROOT / "sample_police_memo.txt", OUT_DIR / "police_memo.pdf"),
    ]
    for t,p in sources:
        if t.exists():
            print(f"Converting {t.name} -> {p.name}")
            try:
                text_to_pdf(t,p)
            except Exception as e:
                print("Failed to convert with reportlab, copying text file instead:", e)
                shutil.copy(str(t), str(p))
        else:
            print(f"Source {t} not found, skipping")

def create_images():
    person_img = OUT_DIR / "person_photo.jpg"
    scene_img = OUT_DIR / "scene_photo.jpg"
    create_placeholder_image(person_img, "Person Photo: John Doe")
    create_placeholder_image(scene_img, "Scene Photo: ABC Supermarket")

def write_metadata():
    metadata = {
        "case_number": "CR-2024-001234",
        "case_id": "CR-2024-001234",
        "location": "MG Road, Bangalore",
        "date_of_incident": "2024-11-15T20:12:00",
        "notes": "Sample case for testing document-first pipeline",
        "uploader": "UnitTest"
    }
    with open(OUT_DIR.parent / "case_metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

def main():
    print("Creating sample case folder under:", OUT_DIR)
    copy_and_convert_text_files()
    create_images()
    write_metadata()
    print("Done.")

if __name__ == '__main__':
    main()
