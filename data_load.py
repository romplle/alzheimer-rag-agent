import pymupdf4llm
from pathlib import Path

INPUT_FOLDER = "data/papers"
OUTPUT_FOLDER = "data/extracted_texts"

Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

pdf_files = list(Path(INPUT_FOLDER).glob("*.pdf"))
print(f"Найдено {len(pdf_files)} pdf-файлов.")

for pdf_path in pdf_files:
    md_text = pymupdf4llm.to_markdown(
        pdf_path,
        pages=None,
        hdr_info=True,
        write_images=False,
    )
    
    output_path = Path(OUTPUT_FOLDER) / f"{pdf_path.stem}.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    
    print(f"Обработан: {pdf_path.name} → {output_path.name} (длина: {len(md_text):,} символов)")
