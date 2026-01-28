import pymupdf.layout
import pymupdf4llm
from pathlib import Path
import re

INPUT_FOLDER = "data/papers"
OUTPUT_FOLDER = "data/extracted_texts"

Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

pdf_files = list(Path(INPUT_FOLDER).glob("*.pdf"))
print(f"Найдено {len(pdf_files)} pdf-файлов.")

def extract_core_sections(text):
    keep = []
    current = None

    sections = {
        "abstract": ["abstract"],
        "introduction": ["introduction"],
        "conclusion": ["conclusion", "discussion", "concluding"]
    }

    for line in text.splitlines():
        l = line.lower().strip()

        if l.startswith(("#", "*")):
            current = None
            for k, keys in sections.items():
                if any(key in l for key in keys):
                    current = k
                    break

        if current:
            keep.append(line)

    return "\n".join(keep)

def remove_figure_captions(text):
    patterns = [
        r'Figure \d+[A-Za-z]*:.*?(?=\n\n|\n[A-Z]|$)',
        r'Fig\. \d+[A-Za-z]*\.?.*?(?=\n\n|\n[A-Z]|$)',
        r'Table \d+[A-Za-z]*:.*?(?=\n\n|\n[A-Z]|$)',
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.MULTILINE | re.DOTALL)
    
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def merge_broken_paragraphs(text):
    lines = text.splitlines()
    merged = []
    
    for line in lines:
        line = line.strip()
        if not line:
            merged.append("")
            continue
        
        if (
            merged
            and merged[-2]
            and not merged[-2].endswith(('.', '!', '?', ':', '*'))
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
        use_ocr=False,
        ignore_images=True,
        ignore_graphics=True,
        force_text=False
    )

    md_text = re.sub(r'(?:\*\*)?==>\s*picture\s*\[.*?\]\s*intentionally\s*omitted\s*<==(?:\*\*)?', '', md_text, flags=re.IGNORECASE)
    md_text = remove_figure_captions(md_text)
    md_text = extract_core_sections(md_text)
    md_text = merge_broken_paragraphs(md_text)
    md_text = re.sub(r'\n{3,}', '\n\n', md_text)
    
    output_path = Path(OUTPUT_FOLDER) / f"{pdf_path.stem}.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    
    print(f"Обработан: {pdf_path.name} → {output_path.name} (длина: {len(md_text):,} символов)")
