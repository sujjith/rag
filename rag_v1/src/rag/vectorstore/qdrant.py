# src/rag/vectorstore/qdrant.py
from qdrant_client import QdrantClient  # Import the Qdrant client to connect to the vector database
from qdrant_client.models import (  # Import specific models we need from Qdrant
    VectorParams, Distance, PointStruct,  # VectorParams: config for vectors, Distance: similarity metric, PointStruct: data point format
    Filter, FieldCondition, MatchValue  # These are used to filter search results by specific conditions
)
from rag.config import settings  # Import our app settings (host, port, collection name, etc.)
from rag.models import Chunk  # Import our Chunk data model

class QdrantStore:
    """Qdrant vector store wrapper."""

    def __init__(self):
        self.client = QdrantClient(  # Create a connection to the Qdrant database
            host=settings.qdrant_host,  # Server address (e.g., "localhost")
            port=settings.qdrant_port  # Port number (typically 6333)
        )
        self.collection_name = settings.collection_name  # Name of the collection where we'll store vectors
        self._ensure_collection()  # Make sure the collection exists when we initialize

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        collections = self.client.get_collections().collections  # Get a list of all existing collections
        exists = any(c.name == self.collection_name for c in collections)  # Check if our collection name is in the list

        if not exists:  # If the collection doesn't exist yet
            self.client.create_collection(  # Create a new collection
                collection_name=self.collection_name,  # Name it with our configured collection name
                vectors_config=VectorParams(  # Configure how vectors should be stored
                    size=settings.embedding_dimension,  # Number of dimensions in each vector (e.g., 384 for all-MiniLM-L6-v2)
                    distance=Distance.COSINE  # Use cosine similarity to measure how similar vectors are (ranges from -1 to 1)
                )
            )

    def upsert(self, chunks: list[Chunk]):
        """Insert or update chunks in Qdrant."""
        points = [  # Build a list of points (data entries) to store in Qdrant
            PointStruct(  # Each point contains an ID, vector, and metadata
                id=chunk.id,  # Unique identifier for this chunk
                vector=chunk.embedding,  # The vector representation (list of numbers) of the chunk's text
                payload={  # Additional data we want to store alongside the vector
                    "document_id": chunk.document_id,  # Which document this chunk came from
                    "content": chunk.content,  # The actual text content of the chunk
                    "metadata": chunk.metadata,  # Extra information (file type, source, etc.)
                    "start_char": chunk.start_char,  # Where this chunk starts in the original document
                    "end_char": chunk.end_char,  # Where this chunk ends in the original document
                }
            )
            for chunk in chunks  # Do this for every chunk in our list
            if chunk.embedding is not None  # Only include chunks that have embeddings (skip if embedding failed)
        ]

        self.client.upsert(  # Insert new points or update existing ones (based on ID)
            collection_name=self.collection_name,  # Which collection to add them to
            points=points  # The list of points we just created
        )

    def search(
        self,
        query_vector: list[float],  # The vector representation of the user's question
        top_k: int = 5,  # How many similar chunks to return (default is 5)
        filter_doc_id: str = None  # Optional: only search within a specific document
    ) -> list[tuple[Chunk, float]]:  # Returns a list of (Chunk, similarity_score) pairs
        """Search for similar chunks."""
        search_filter = None  # Start with no filter
        if filter_doc_id:  # If the user wants to filter by document ID
            search_filter = Filter(  # Create a filter object
                must=[FieldCondition(  # The "must" means this condition is required
                    key="document_id",  # We're filtering on the document_id field
                    match=MatchValue(value=filter_doc_id)  # It must match this specific document ID
                )]
            )

        results = self.client.query_points(  # Query for similar vectors using the new Qdrant API
            collection_name=self.collection_name,  # Which collection to search in
            query=query_vector,  # The vector we're comparing against (the user's question)
            limit=top_k,  # Maximum number of results to return
            query_filter=search_filter,  # Apply the filter if one was specified (or None)
            with_payload=True,  # Include the stored payload data in results
        ).points  # Extract the points from the response

        chunks = []  # Create an empty list to store our results
        for result in results:  # Loop through each search result from Qdrant
            chunk = Chunk(  # Reconstruct a Chunk object from the stored data
                id=str(result.id),  # The chunk's unique ID (convert to string)
                document_id=result.payload["document_id"],  # Extract document_id from the stored payload
                content=result.payload["content"],  # Extract the text content
                metadata=result.payload["metadata"],  # Extract the metadata
                start_char=result.payload["start_char"],  # Extract start position
                end_char=result.payload["end_char"],  # Extract end position
            )
            chunks.append((chunk, result.score))  # Add a tuple of (Chunk object, similarity score) to our list

        return chunks  # Return the list of (chunk, score) pairs