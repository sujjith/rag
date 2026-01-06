# src/rag/ingestion/chunker.py
import re  # Regular expressions for splitting text
from typing import Generator  # Type hint for functions that yield values
from rag.models import Document, Chunk  # Import our data models
import uuid  # For generating UU IDs (Qdrant-compatible)

class SentenceChunker:  # Splits documents into smaller chunks by sentence boundaries

    def __init__(self, chunk_size: int = 512, overlap: int = 50):  # Initialize with chunk settings
        self.chunk_size = chunk_size  # Max characters per chunk (default 512)
        self.overlap = overlap  # Characters to repeat between chunks (maintains context)
        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+')  # Regex: split after . ! ? followed by whitespace

    def chunk(self, document: Document) -> Generator[Chunk, None, None]:  # Generator yields chunks one at a time
        sentences = self.sentence_pattern.split(document.content)  # Split document into sentences

        current_chunk = []  # List of sentences in current chunk
        current_length = 0  # Total character count in current chunk
        start_char = 0  # Position in original document where current chunk starts

        for sentence in sentences:  # Process each sentence
            sentence_len = len(sentence)  # Get length of current sentence

            if current_length + sentence_len > self.chunk_size and current_chunk:  # If adding sentence exceeds limit
                # Yield current chunk
                chunk_text = " ".join(current_chunk)  # Combine sentences into chunk text
                chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{document.id}:{start_char}"))  # Generate UUID from document ID + position

                yield Chunk(  # Yield = return one chunk, then continue (generator pattern)
                    id=chunk_id,
                    document_id=document.id,
                    content=chunk_text,
                    start_char=start_char,
                    end_char=start_char + len(chunk_text),
                    metadata=document.metadata.copy()  # Copy metadata to each chunk
                )

                overlap_text = chunk_text[-self.overlap:] if len(chunk_text) > self.overlap else chunk_text  # Get last N chars for overlap
                current_chunk = [overlap_text]  # Start new chunk with overlap text
                current_length = len(overlap_text)  # Reset length counter
                start_char += len(chunk_text) - len(overlap_text)  # Move start position forward

            current_chunk.append(sentence)  # Add sentence to current chunk
            current_length += sentence_len  # Update character count

        if current_chunk:  # Handle any remaining sentences after loop ends
            chunk_text = " ".join(current_chunk)  # Combine remaining sentences
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{document.id}:{start_char}"))  # Generate UUID for final chunk

            yield Chunk(  # Yield the final chunk
                id=chunk_id,
                document_id=document.id,
                content=chunk_text,
                start_char=start_char,
                end_char=start_char + len(chunk_text),
                metadata=document.metadata.copy()
            )

