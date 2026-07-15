"""Document chunking with semantic splitting and sliding window overlap."""

import re
import logging
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)


def semantic_chunk(
    text: str,
    page_number: int = 1,
    chunk_size: int = None,
    chunk_overlap: int = None,
    metadata: dict = None
) -> list[dict]:
    """
    Chunk text using semantic boundaries (paragraphs, headers) with 
    sliding window overlap as fallback for large sections.
    
    Returns list of dicts with 'content', 'page_number', 'line_number', and any extra metadata.
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    metadata = metadata or {}

    # Split by semantic boundaries: double newlines, markdown headers, horizontal rules
    sections = re.split(r'\n{2,}|(?=^#{1,6}\s)', text, flags=re.MULTILINE)
    sections = [s.strip() for s in sections if s.strip()]

    chunks = []
    current_chunk = ""

    for section in sections:
        # If adding this section would exceed chunk_size, finalize current chunk
        if current_chunk and len(current_chunk) + len(section) + 1 > chunk_size:
            line_num = text[:text.find(current_chunk)].count('\n') + 1 if current_chunk in text else 1
            chunks.append(_make_chunk(current_chunk, page_number, line_num, metadata))
            # Sliding window overlap: keep the tail of the current chunk, aligning to word boundary
            if chunk_overlap > 0:
                overlap_target = max(0, len(current_chunk) - chunk_overlap)
                aligned_start = current_chunk.find(' ', overlap_target)
                if aligned_start != -1:
                    overlap_text = current_chunk[aligned_start + 1:].strip()
                else:
                    overlap_text = current_chunk[overlap_target:].strip()
            else:
                overlap_text = ""
            current_chunk = overlap_text + " " + section if overlap_text else section
        else:
            current_chunk = current_chunk + "\n\n" + section if current_chunk else section

        # If a single section is larger than chunk_size, split it further
        while len(current_chunk) > chunk_size:
            # Try to split at sentence boundary
            split_pos = _find_sentence_boundary(current_chunk, chunk_size)
            chunk_text = current_chunk[:split_pos].strip()
            if chunk_text:
                line_num = text[:text.find(chunk_text)].count('\n') + 1 if chunk_text in text else 1
                chunks.append(_make_chunk(chunk_text, page_number, line_num, metadata))
            
            if chunk_overlap > 0:
                overlap_target = max(0, split_pos - chunk_overlap)
                aligned_start = current_chunk.find(' ', overlap_target)
                if aligned_start != -1 and aligned_start < split_pos:
                    overlap_start = aligned_start + 1
                else:
                    overlap_start = overlap_target
            else:
                overlap_start = split_pos
                
            current_chunk = current_chunk[overlap_start:].strip()

    # Don't forget the last chunk
    if current_chunk.strip():
        line_num = text[:text.find(current_chunk.strip())].count('\n') + 1 if current_chunk.strip() in text else 1
        chunks.append(_make_chunk(current_chunk.strip(), page_number, line_num, metadata))

    return chunks


def _find_sentence_boundary(text: str, max_pos: int) -> int:
    """Find the best sentence boundary before max_pos."""
    # Look for sentence-ending punctuation followed by space or newline
    candidates = []
    for match in re.finditer(r'[.!?][\s\n]', text[:max_pos]):
        candidates.append(match.end())
    
    if candidates:
        return candidates[-1]  # Last sentence boundary before max_pos
    
    # Fallback: split at last newline or space
    last_newline = text[:max_pos].rfind('\n')
    if last_newline > 0:
        return last_newline + 1
        
    last_space = text[:max_pos].rfind(' ')
    if last_space > 0:
        return last_space + 1
    
    # Last resort: hard split
    return max_pos


def _make_chunk(content: str, page_number: int, line_number: int, metadata: dict) -> dict:
    """Create a chunk dictionary with content and metadata."""
    return {
        "content": content,
        "page_number": page_number,
        "line_number": line_number,
        **metadata
    }


def chunk_document_pages(pages: list[dict], metadata: dict = None) -> list[dict]:
    """
    Chunk a full document given as a list of page dicts.
    Each page dict should have 'text' and 'page_number'.
    Returns a flat list of chunk dicts.
    """
    all_chunks = []
    for page in pages:
        page_chunks = semantic_chunk(
            text=page["text"],
            page_number=page.get("page_number", 1),
            metadata=metadata or {}
        )
        all_chunks.extend(page_chunks)
    
    logger.info(f"Chunked document into {len(all_chunks)} chunks from {len(pages)} pages")
    return all_chunks
