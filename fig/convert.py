# pdf_to_png.py
from pathlib import Path
import fitz  # PyMuPDF

pdf_path = Path("Figure1_new.pdf")
output_png = pdf_path.with_suffix(".png")

with fitz.open(pdf_path) as doc:
    page = doc.load_page(0)           # Load the first page (index starts at 0)
    pix = page.get_pixmap(dpi=300)    # 300 DPI is typically sharp enough
    pix.save(output_png)

print(f"Converted to: {output_png}")