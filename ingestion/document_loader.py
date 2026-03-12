"""
ingestion/document_loader.py — Document Loading
================================================
WHY THIS EXISTS:
    Financial data comes in many formats — PDFs, news HTML,
    plain text. This module handles ALL of them in one place.
    The rest of the system doesn't care about the source format,
    it just receives clean text.

WHAT YOU LEARN:
    - How to extract text from PDFs
    - How to scrape financial news HTML
    - How to build modular, extensible loaders
    - Error handling in data pipelines
"""

import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from pypdf import PdfReader
from bs4 import BeautifulSoup
import requests

from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class Document:
    """
    Represents a single loaded document.

    WHY A DATACLASS?
    Instead of passing raw strings around, we wrap content
    in a structured object. This makes the pipeline cleaner
    and easier to debug — you always know what a Document
    contains.
    """
    content: str           # The actual text
    source: str            # Where it came from (file path or URL)
    doc_type: str          # "pdf", "html", "text"
    metadata: dict         # Extra info (page count, title, date, etc.)


class DocumentLoader:
    """
    Loads financial documents from multiple sources.

    Supported formats:
        - PDF files (annual reports, research papers)
        - HTML pages (financial news, web articles)
        - Plain text files (.txt)

    Usage:
        loader = DocumentLoader()
        doc = loader.load_pdf("data/raw/apple_10k.pdf")
        doc = loader.load_url("https://finance.yahoo.com/news/...")
        docs = loader.load_directory("data/raw/")
    """

    def __init__(self):
        log.info("DocumentLoader initialized")

    def load_pdf(self, file_path: str) -> Optional[Document]:
        """
        Extract text from a PDF file.

        WHY PYPDF?
        It's lightweight and works without any system dependencies.
        Alternatives like pdfplumber are more accurate but heavier.
        For financial reports, pypdf is good enough.

        Args:
            file_path: Path to the PDF file

        Returns:
            Document object with extracted text, or None if failed
        """
        path = Path(file_path)

        if not path.exists():
            log.error(f"PDF file not found: {file_path}")
            return None

        try:
            reader = PdfReader(str(path))
            pages_text = []

            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    # Clean the extracted text
                    text = self._clean_text(text)
                    if text.strip():
                        pages_text.append(text)

            full_text = "\n\n".join(pages_text)

            if not full_text.strip():
                log.warning(f"No text extracted from PDF: {file_path}")
                return None

            doc = Document(
                content=full_text,
                source=str(path),
                doc_type="pdf",
                metadata={
                    "filename": path.name,
                    "page_count": len(reader.pages),
                    "file_size_kb": round(path.stat().st_size / 1024, 2)
                }
            )

            log.info(
                f"Loaded PDF: {path.name} | "
                f"{len(reader.pages)} pages | "
                f"{len(full_text)} characters"
            )
            return doc

        except Exception as e:
            log.error(f"Failed to load PDF {file_path}: {str(e)}")
            return None

    def load_url(self, url: str) -> Optional[Document]:
        """
        Load and extract text from a financial news URL.

        WHY BEAUTIFULSOUP?
        HTML pages have lots of noise — nav bars, ads, footers.
        BeautifulSoup lets us extract just the article content.

        Args:
            url: Full URL of the news article or financial page

        Returns:
            Document object with extracted text, or None if failed
        """
        try:
            headers = {
                # Pretend to be a browser — some sites block Python requests
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/91.0.4472.124 Safari/537.36"
                )
            }

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()  # Raises error for 404, 500, etc.

            soup = BeautifulSoup(response.content, "html.parser")

            # Remove noise elements from HTML
            # These tags never contain useful article content
            for tag in soup(["script", "style", "nav", "footer",
                              "header", "aside", "advertisement"]):
                tag.decompose()

            # Try to find the main article content
            # Different news sites use different HTML structures
            article = (
                soup.find("article") or
                soup.find("main") or
                soup.find("div", class_=re.compile(r"article|content|story"))
            )

            if article:
                text = article.get_text(separator="\n")
            else:
                # Fallback — get all paragraph text
                paragraphs = soup.find_all("p")
                text = "\n".join(p.get_text() for p in paragraphs)

            text = self._clean_text(text)

            if not text.strip():
                log.warning(f"No text extracted from URL: {url}")
                return None

            doc = Document(
                content=text,
                source=url,
                doc_type="html",
                metadata={
                    "url": url,
                    "title": soup.title.string if soup.title else "Unknown",
                    "char_count": len(text)
                }
            )

            log.info(f"Loaded URL: {url[:60]}... | {len(text)} characters")
            return doc

        except requests.exceptions.Timeout:
            log.error(f"Timeout loading URL: {url}")
            return None
        except requests.exceptions.RequestException as e:
            log.error(f"Failed to load URL {url}: {str(e)}")
            return None

    def load_text_file(self, file_path: str) -> Optional[Document]:
        """
        Load a plain text file.

        Args:
            file_path: Path to .txt file

        Returns:
            Document object or None if failed
        """
        path = Path(file_path)

        if not path.exists():
            log.error(f"Text file not found: {file_path}")
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            text = self._clean_text(text)

            doc = Document(
                content=text,
                source=str(path),
                doc_type="text",
                metadata={
                    "filename": path.name,
                    "char_count": len(text)
                }
            )

            log.info(f"Loaded text file: {path.name} | {len(text)} characters")
            return doc

        except Exception as e:
            log.error(f"Failed to load text file {file_path}: {str(e)}")
            return None

    def load_directory(self, directory: str) -> list[Document]:
        """
        Load all supported documents from a directory.

        This is useful for batch ingestion — point it at a folder
        of PDFs and it loads them all automatically.

        Args:
            directory: Path to folder containing documents

        Returns:
            List of Document objects (failed files are skipped)
        """
        dir_path = Path(directory)

        if not dir_path.exists():
            log.error(f"Directory not found: {directory}")
            return []

        documents = []

        # Map file extensions to loader methods
        loaders = {
            ".pdf": self.load_pdf,
            ".txt": self.load_text_file,
        }

        all_files = list(dir_path.iterdir())
        log.info(f"Scanning directory: {directory} | {len(all_files)} files found")

        for file_path in all_files:
            if file_path.suffix.lower() in loaders:
                loader_fn = loaders[file_path.suffix.lower()]
                doc = loader_fn(str(file_path))
                if doc:
                    documents.append(doc)
            else:
                log.debug(f"Skipping unsupported file: {file_path.name}")

        log.info(
            f"Directory loading complete | "
            f"{len(documents)} documents loaded successfully"
        )
        return documents

    def _clean_text(self, text: str) -> str:
        """
        Clean raw extracted text.

        Financial PDFs often have messy text:
        - Multiple spaces and newlines
        - Page numbers like "Page 12 of 45"
        - Headers/footers repeated on every page
        - Strange unicode characters

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text string
        """
        if not text:
            return ""

        # Remove page number patterns (e.g., "Page 1 of 10", "- 12 -")
        text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'-\s*\d+\s*-', '', text)

        # Replace multiple whitespace/newlines with single ones
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)

        # Remove non-printable characters
        text = re.sub(r'[^\x20-\x7E\n]', ' ', text)

        return text.strip()