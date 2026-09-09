import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pymupdf as fitz
import pandas as pd
import pdfplumber
import pytesseract
from PIL import Image
from langchain_core.documents import Document

from exception.custom_exception import DocumentPortalException
from logger.custom_logger import CustomLogger


logger = CustomLogger().get_logger(__name__)



class ComplexPDFParser:
    """
    Parse a complex PDF containing:
    - selectable text
    - scanned pages
    - tables
    - embedded images

    Final output can be converted into LangChain Documents for RAG.
    """

    def __init__(
        self,
        pdf_path: str,
        output_dir: str,
        tesseract_path: Optional[str] = None,
        render_scale: float = 2.0,
    ):
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path(output_dir)
        self.image_dir = self.output_dir / "extracted_images"
        self.page_image_dir = self.output_dir / "page_images"
        self.render_scale = render_scale

        self.page_records: List[Dict[str, Any]] = []
        self.image_records: List[Dict[str, Any]] = []
        self.table_records: List[Dict[str, Any]] = []
        self.langchain_docs: List[Document] = []

        try:
            self._validate_pdf()
            self._create_output_directories()
            self._configure_tesseract(tesseract_path)
            logger.info(
                "pdf_parser_initialized",
                pdf_path=str(self.pdf_path),
                output_dir=str(self.output_dir),
            )
        except Exception as exc:
            logger.exception(
                "pdf_parser_initialization_failed",
                pdf_path=str(self.pdf_path),
                error=str(exc),
            )
            raise DocumentPortalException(
                "Failed to initialize the PDF parser",
                exc,
            ) from exc

    # ============================================================
    # Setup
    # ============================================================

    def _validate_pdf(self) -> None:
        """Validate that the input PDF exists."""
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")

    def _create_output_directories(self) -> None:
        """Create directories used by the parser."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.page_image_dir.mkdir(parents=True, exist_ok=True)

    def _configure_tesseract(self, tesseract_path: Optional[str]) -> None:
        """
        Configure Tesseract executable.

        If a path is provided, it is validated and assigned to pytesseract.
        Otherwise, discover Tesseract from PATH or its standard Windows
        installation locations.
        """
        if tesseract_path:
            candidates = [Path(tesseract_path)]
        else:
            candidates = []
            path_from_environment = os.getenv("TESSERACT_PATH")
            if path_from_environment:
                candidates.append(Path(path_from_environment))

            path_from_path = shutil.which("tesseract")
            if path_from_path:
                candidates.append(Path(path_from_path))

            if os.name == "nt":
                candidates.extend(
                    [
                        Path(os.environ.get("ProgramFiles", "C:\\Program Files"))
                        / "Tesseract-OCR"
                        / "tesseract.exe",
                        Path(os.environ.get("LOCALAPPDATA", ""))
                        / "Tesseract-OCR"
                        / "tesseract.exe",
                    ]
                )

        path = next((candidate for candidate in candidates if candidate.is_file()), None)

        if path is None:
            if tesseract_path:
                raise FileNotFoundError(
                    f"Tesseract executable not found at: {tesseract_path}"
                )
            return

        pytesseract.pytesseract.tesseract_cmd = str(path)

    # ============================================================
    # OCR
    # ============================================================

    def run_ocr_on_image(self, image_path: str | Path) -> str:
        """
        Run OCR on an image using pytesseract.

        Returns extracted text.
        If OCR fails, returns a readable failure marker instead
        of stopping the entire PDF parsing pipeline.
        """
        try:
            with Image.open(image_path) as image:
                text = pytesseract.image_to_string(image)

            return text.strip()

        except Exception as exc:
            logger.warning(
                "image_ocr_failed",
                image_path=str(image_path),
                error=str(exc),
            )
            return f"[OCR_SKIPPED_OR_FAILED: {exc}]"

    # ============================================================
    # PyMuPDF: Text + Images + Full-page OCR
    # ============================================================

    def extract_text_and_images(
        self,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extract:
        - selectable page text
        - page dimensions
        - full-page rendered images
        - OCR text from rendered pages
        - embedded images
        - OCR text from embedded images
        """
        page_records: List[Dict[str, Any]] = []
        image_records: List[Dict[str, Any]] = []

        pdf = fitz.open(str(self.pdf_path))

        try:
            for page_index in range(len(pdf)):
                page = pdf[page_index]
                page_number = page_index + 1

                images = page.get_images(full=True)
                selectable_text = page.get_text("text").strip()

                # Render full page for OCR
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(self.render_scale, self.render_scale)
                )

                page_image_path = (
                    self.page_image_dir / f"page_{page_number:03d}.png"
                )
                pixmap.save(str(page_image_path))

                page_ocr_text = self.run_ocr_on_image(page_image_path)

                page_info = {
                    "page_number": page_number,
                    "text": selectable_text,
                    "ocr_text": page_ocr_text,
                    "image_count": len(images),
                    "width": page.rect.width,
                    "height": page.rect.height,
                    "page_image_path": str(page_image_path),
                }

                page_records.append(page_info)

                # Extract embedded images
                for image_index, image_info in enumerate(images, start=1):
                    xref = image_info[0]
                    base_image = pdf.extract_image(xref)

                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    image_path = (
                        self.image_dir
                        / (
                            f"page_{page_number:03d}_"
                            f"image_{image_index}.{image_ext}"
                        )
                    )

                    with open(image_path, "wb") as file:
                        file.write(image_bytes)

                    image_ocr_text = self.run_ocr_on_image(image_path)

                    image_records.append(
                        {
                            "page_number": page_number,
                            "image_index": image_index,
                            "image_path": str(image_path),
                            "image_ext": image_ext,
                            "image_ocr_text": image_ocr_text,
                        }
                    )

        finally:
            pdf.close()

        self.page_records = page_records
        self.image_records = image_records

        return page_records, image_records

    # ============================================================
    # pdfplumber: Tables
    # ============================================================

    def extract_tables(self) -> List[Dict[str, Any]]:
        """Extract tables from every page using pdfplumber."""
        table_records: List[Dict[str, Any]] = []

        with pdfplumber.open(str(self.pdf_path)) as pdf:
            for page_index, page in enumerate(pdf.pages):
                page_number = page_index + 1

                try:
                    tables = page.extract_tables() or []
                except Exception as exc:
                    logger.warning(
                        "table_extraction_failed",
                        pdf_path=str(self.pdf_path),
                        page_number=page_number,
                        error=str(exc),
                    )
                    tables = []

                for table_index, table in enumerate(tables, start=1):
                    if not table:
                        continue

                    cleaned_table = []

                    for row in table:
                        cleaned_row = [
                            cell.strip() if isinstance(cell, str) else cell
                            for cell in row
                        ]
                        cleaned_table.append(cleaned_row)

                    if not cleaned_table:
                        continue

                    try:
                        dataframe = pd.DataFrame(
                            cleaned_table[1:],
                            columns=cleaned_table[0],
                        )
                    except Exception:
                        dataframe = pd.DataFrame(cleaned_table)

                    table_records.append(
                        {
                            "page_number": page_number,
                            "table_index": table_index,
                            "raw_table": cleaned_table,
                            "markdown": dataframe.to_markdown(index=False),
                            "csv": dataframe.to_csv(index=False),
                        }
                    )

        self.table_records = table_records
        return table_records

    # ============================================================
    # LangChain Documents
    # ============================================================

    def create_langchain_documents(self) -> List[Document]:
        """
        Convert parsed page text, OCR, tables and image OCR
        into LangChain Document objects.
        """
        documents: List[Document] = []

        # Page text + page OCR
        for page in self.page_records:
            page_number = page["page_number"]

            combined_text = f"""
PAGE {page_number}

SELECTABLE TEXT:
{page["text"]}

OCR TEXT:
{page["ocr_text"]}
""".strip()

            documents.append(
                Document(
                    page_content=combined_text,
                    metadata={
                        "source": str(self.pdf_path),
                        "page_number": page_number,
                        "content_type": "page_text_plus_ocr",
                        "image_count": page["image_count"],
                        "page_image_path": page["page_image_path"],
                    },
                )
            )

        # Tables
        for table in self.table_records:
            page_number = table["page_number"]

            table_text = f"""
TABLE FOUND ON PAGE {page_number}
TABLE INDEX: {table["table_index"]}

TABLE MARKDOWN:
{table["markdown"]}
""".strip()

            documents.append(
                Document(
                    page_content=table_text,
                    metadata={
                        "source": str(self.pdf_path),
                        "page_number": page_number,
                        "content_type": "table",
                        "table_index": table["table_index"],
                    },
                )
            )

        # Embedded image OCR
        for image in self.image_records:
            page_number = image["page_number"]

            image_text = f"""
IMAGE FOUND ON PAGE {page_number}
IMAGE INDEX: {image["image_index"]}
IMAGE PATH: {image["image_path"]}

IMAGE OCR TEXT:
{image["image_ocr_text"]}
""".strip()

            documents.append(
                Document(
                    page_content=image_text,
                    metadata={
                        "source": str(self.pdf_path),
                        "page_number": page_number,
                        "content_type": "image",
                        "image_index": image["image_index"],
                        "image_path": image["image_path"],
                        "image_ext": image["image_ext"],
                    },
                )
            )

        self.langchain_docs = documents
        return documents

    # ============================================================
    # Save Outputs
    # ============================================================

    @staticmethod
    def _save_json(path: Path, data: Any) -> None:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

    def save_outputs(self) -> None:
        """Save parsed records and RAG-ready documents to disk."""
        self._save_json(
            self.output_dir / "page_records.json",
            self.page_records,
        )

        self._save_json(
            self.output_dir / "image_records.json",
            self.image_records,
        )

        self._save_json(
            self.output_dir / "table_records.json",
            self.table_records,
        )

        # All LangChain Documents as Markdown
        rag_documents_path = self.output_dir / "rag_ready_documents.md"

        with open(rag_documents_path, "w", encoding="utf-8") as file:
            for index, document in enumerate(
                self.langchain_docs,
                start=1,
            ):
                file.write(f"\n\n# Document {index}\n")
                file.write("\nMetadata:\n```json\n")
                file.write(
                    json.dumps(
                        document.metadata,
                        indent=2,
                        ensure_ascii=False,
                        default=str,
                    )
                )
                file.write("\n```\n")
                file.write("\nContent:\n")
                file.write(document.page_content)
                file.write("\n\n---\n")

        # Tables as Markdown
        tables_path = self.output_dir / "extracted_tables.md"

        with open(tables_path, "w", encoding="utf-8") as file:
            for table in self.table_records:
                file.write(
                    f"\n\n## Page {table['page_number']} "
                    f"- Table {table['table_index']}\n\n"
                )
                file.write(table["markdown"])
                file.write("\n\n---\n")

    # ============================================================
    # Main Pipeline
    # ============================================================

    def parse(self, save_output: bool = True) -> Dict[str, Any]:
        """
        Run the complete parsing pipeline.

        Returns:
            {
                "pages": [...],
                "images": [...],
                "tables": [...],
                "documents": [LangChain Document, ...],
                "output_dir": "..."
            }
        """
        logger.info(
            "pdf_parsing_started",
            pdf_path=str(self.pdf_path),
        )

        try:
            self.extract_text_and_images()
            logger.info(
                "pdf_content_extracted",
                pages=len(self.page_records),
                images=len(self.image_records),
            )

            self.extract_tables()
            logger.info(
                "pdf_tables_extracted",
                tables=len(self.table_records),
            )

            self.create_langchain_documents()
            logger.info(
                "langchain_documents_created",
                documents=len(self.langchain_docs),
            )

            if save_output:
                self.save_outputs()
                logger.info(
                    "parser_outputs_saved",
                    output_dir=str(self.output_dir),
                )

            result = {
                "pages": self.page_records,
                "images": self.image_records,
                "tables": self.table_records,
                "documents": self.langchain_docs,
                "output_dir": str(self.output_dir),
            }

            logger.info(
                "pdf_parsing_completed",
                pdf_path=str(self.pdf_path),
            )
            return result

        except DocumentPortalException:
            raise
        except Exception as exc:
            logger.exception(
                "pdf_parsing_failed",
                pdf_path=str(self.pdf_path),
                error=str(exc),
            )
            raise DocumentPortalException(
                f"Failed to parse PDF: {self.pdf_path}",
                exc,
            ) from exc

# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":

    DATA_DIR = PROJECT_ROOT / "data"

    PDF_PATH = DATA_DIR / "Client_Contracts_Policies_and_Incident_Records.pdf"
    OUTPUT_DIR = DATA_DIR / "parsed_pdf_output"

    TESSERACT_PATH = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )

    parser = ComplexPDFParser(
        pdf_path=PDF_PATH,
        output_dir=OUTPUT_DIR,
        tesseract_path=TESSERACT_PATH,
    )

    result = parser.parse()

    documents = result["documents"]

    print("\n================ SUMMARY ================\n")
    print("Pages:", len(result["pages"]))
    print("Images:", len(result["images"]))
    print("Tables:", len(result["tables"]))
    print("LangChain Documents:", len(documents))

    if documents:
        print("\nFirst document preview:\n")
        print(documents[0].page_content[:1500])
