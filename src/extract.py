from pathlib import Path

import docx
import pandas as pd
import pypdf
import tabula

import src.params as params


def _extract_tables_from_docx(table):
    data = []
    keys = None
    for i, row in enumerate(table.rows):
        text = (cell.text for cell in row.cells)

        if i == 0:
            keys = tuple(text)
            continue
        row_data = dict(zip(keys, text))
        data.append(row_data)
    return pd.DataFrame(data).to_markdown()


def _extract_tables_from_pdf(file_path: Path):
    tables = tabula.read_pdf(file_path, pages="all")
    return [table.to_markdown() for table in tables]


def extract_text_from_document(file_path: Path):
    if file_path.suffix == ".pdf":
        pdf = pypdf.PdfReader(open(file_path, "rb"))
        text = "\n\n".join([page.extract_text() for page in pdf.pages])
        tables = "\n\n".join(_extract_tables_from_pdf(file_path))

    elif file_path.suffix.startswith(".doc"):
        doc = docx.Document(file_path)
        text = "\n\n".join([para.text for para in doc.paragraphs])
        tables = "\n\n".join([_extract_tables_from_docx(table) for table in doc.tables])

    return f"Document Text: \n\n {text}\n\n Tables: \n\n {tables}"


def main():
    params.TXT_DIR.mkdir(exist_ok=True, parents=True)

    for file in params.PDF_DIR.iterdir():
        out_file = params.TXT_DIR / file.with_suffix(".txt").name
        if out_file.exists():
            continue
        text = extract_text_from_document(file)
        with open(out_file, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
