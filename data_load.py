import pymupdf.layout
import pymupdf4llm
from pathlib import Path

INPUT_FOLDER = "data/papers"
OUTPUT_FOLDER = "data/extracted_texts"

Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

pdf_files = list(Path(INPUT_FOLDER).glob("*.pdf"))
print(f"Найдено {len(pdf_files)} pdf-файлов.")

def merge_broken_paragraphs(md: str) -> str:
    lines = md.splitlines()
    merged = []
    
    for line in lines:
        line = line.strip()
        if not line:
            merged.append("")
            continue
        
        if (
            merged
            and merged[-2]
            and not merged[-2].endswith(('.', '!', '?', ':'))
            and line[0].islower()
        ):
            merged[-2] += " " + line
        else:
            merged.append(line)
    
    return "\n".join(merged)

for pdf_path in pdf_files:
    md_text = pymupdf4llm.to_markdown(
        pdf_path,
        pages=None,
        hdr_info=False,
        write_images=False,
        footer=False,
        header=False,
        margins=50,
        use_ocr=False
    )

    md_text = merge_broken_paragraphs(md_text)
    
    output_path = Path(OUTPUT_FOLDER) / f"{pdf_path.stem}.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    
    print(f"Обработан: {pdf_path.name} → {output_path.name} (длина: {len(md_text):,} символов)")
