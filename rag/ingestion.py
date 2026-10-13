import gc
import hashlib
import os
from pathlib import Path

import chromadb
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from config.settings import (
    CHROMA_PERSIST_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAMES,
    DATA_DIRS,
)
from rag.embeddings import embed_texts

_chroma_client = None


def get_chroma_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return _chroma_client


def get_or_create_collection(source_type: str) -> chromadb.Collection:
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAMES[source_type],
        metadata={"hnsw:space": "cosine"},
    )


def _build_converter() -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    # Disabled: image generation is very RAM-intensive on large PDFs
    pipeline_options.generate_page_images = False
    pipeline_options.generate_picture_images = False
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if not text.strip():
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def _chunk_id(source_type: str, file_name: str, page: int, idx: int, chunk_type: str) -> str:
    raw = f"{source_type}::{file_name}::{page}::{idx}::{chunk_type}"
    return hashlib.md5(raw.encode()).hexdigest()


def ingest_file(pdf_path: str, source_type: str, converter: DocumentConverter) -> int:
    collection = get_or_create_collection(source_type)
    file_name = Path(pdf_path).name

    print(f"  Parsing {file_name}...")
    try:
        result = converter.convert(pdf_path)
        doc = result.document
    except Exception as e:
        print(f"  ERROR parsing {file_name}: {e}")
        return 0

    documents, metadatas, ids = [], [], []

    # Extract text chunks per page
    for page_no, page in doc.pages.items():
        page_texts = []
        for item in doc.texts:
            if hasattr(item, "prov") and item.prov:
                for prov in item.prov:
                    if prov.page_no == page_no:
                        page_texts.append(item.text)
                        break

        full_text = " ".join(page_texts)
        for idx, chunk in enumerate(_chunk_text(full_text)):
            cid = _chunk_id(source_type, file_name, page_no, idx, "text")
            documents.append(chunk)
            metadatas.append({
                "source_type": source_type,
                "file_name": file_name,
                "page_number": page_no,
                "chunk_type": "text",
                "image_data": "",
            })
            ids.append(cid)

    # Extract tables
    for table in doc.tables:
        try:
            table_md = table.export_to_markdown(doc=doc)
        except TypeError:
            table_md = table.export_to_markdown()
        if not table_md.strip():
            continue
        page_no = table.prov[0].page_no if table.prov else 0
        idx = len(documents)
        cid = _chunk_id(source_type, file_name, page_no, idx, "table")
        documents.append(f"[TABLE]\n{table_md}")
        metadatas.append({
            "source_type": source_type,
            "file_name": file_name,
            "page_number": page_no,
            "chunk_type": "table",
            "image_data": "",
        })
        ids.append(cid)

    # Extract picture captions (no heavy image data — images disabled to save memory)
    for pic in doc.pictures:
        caption = ""
        if pic.captions:
            caption = " ".join(c.text for c in pic.captions if hasattr(c, "text"))
        if not caption:
            continue  # skip uncaptioned images — no useful text to embed
        page_no = pic.prov[0].page_no if pic.prov else 0
        idx = len(documents)
        cid = _chunk_id(source_type, file_name, page_no, idx, "image")
        documents.append(f"[IMAGE CAPTION] {caption}")
        metadatas.append({
            "source_type": source_type,
            "file_name": file_name,
            "page_number": page_no,
            "chunk_type": "image",
            "image_data": "",
        })
        ids.append(cid)

    if not documents:
        print(f"  Warning: no content extracted from {file_name}")
        return 0

    # Skip already-ingested chunks
    existing = set(collection.get(ids=ids)["ids"])
    new_mask = [i for i, cid in enumerate(ids) if cid not in existing]
    if not new_mask:
        print(f"  Already ingested: {file_name} ({len(ids)} chunks)")
        return 0

    batch_docs = [documents[i] for i in new_mask]
    batch_meta = [metadatas[i] for i in new_mask]
    batch_ids = [ids[i] for i in new_mask]

    print(f"  Embedding {len(batch_docs)} chunks from {file_name}...")
    batch_embs = embed_texts(batch_docs)

    BATCH = 200
    for start in range(0, len(batch_docs), BATCH):
        collection.add(
            documents=batch_docs[start:start + BATCH],
            embeddings=batch_embs[start:start + BATCH],
            metadatas=batch_meta[start:start + BATCH],
            ids=batch_ids[start:start + BATCH],
        )

    print(f"  Ingested {len(batch_docs)} new chunks from {file_name}")
    return len(batch_docs)


def ingest_all() -> None:
    converter = _build_converter()
    total = 0
    for source_type, data_dir in DATA_DIRS.items():
        pdfs = list(Path(data_dir).glob("*.pdf"))
        if not pdfs:
            print(f"[{source_type.upper()}] No PDFs found in {data_dir}")
            continue
        print(f"\n[{source_type.upper()}] Ingesting {len(pdfs)} files...")
        for pdf_path in pdfs:
            count = ingest_file(str(pdf_path), source_type, converter)
            total += count
            # Free memory between files
            gc.collect()

    print(f"\nIngestion complete. Total new chunks: {total}")


if __name__ == "__main__":
    ingest_all()
