## 10.5 Advanced RAG Techniques

This section covers next-generation RAG approaches that go beyond traditional retrieve-and-generate.

---

### 10.5.1 Agentic RAG (ReAct Pattern)

**Concept:** The LLM decides *when* to retrieve and can use tools/functions, rather than always retrieving upfront.

**ReAct (Reasoning + Acting) Pattern:**

```python
from typing import List, Dict, Callable
import re

class AgenticRAG:
    """LLM agent that decides when to retrieve documents."""
    
    def __init__(self, llm_client, search_function: Callable, tools: Dict[str, Callable]):
        self.llm = llm_client
        self.search = search_function
        self.tools = tools
        self.max_iterations = 5
    
    def query(self, question: str) -> str:
        """Run agentic RAG with ReAct loop."""
        
        conversation_history = []
        
        # System prompt with tool descriptions
        system_prompt = """You are a helpful assistant with access to tools.

Available tools:
- search(query: str) -> List[str]: Search the knowledge base
- calculator(expression: str) -> float: Evaluate math expressions
- get_current_date() -> str: Get today's date

To use a tool, write:
Thought: I need to search for information about X
Action: search("information about X")

After seeing the Observation, continue reasoning or provide Final Answer.

Always end with:
Final Answer: <your answer>
"""
        
        for iteration in range(self.max_iterations):
            # Build prompt with history
            prompt = system_prompt + "\n\nQuestion: " + question + "\n\n"
            for entry in conversation_history:
                prompt += entry + "\n"
            
            # Get LLM response
            response = self.llm.generate(prompt, max_tokens=300)
            conversation_history.append(response)
            
            # Check if final answer
            if "Final Answer:" in response:
                final_answer = response.split("Final Answer:")[-1].strip()
                return final_answer
            
            # Parse action
            action_match = re.search(r'Action: (\w+)\((.*?)\)', response)
            if not action_match:
                continue
            
            tool_name = action_match.group(1)
            tool_args = action_match.group(2).strip('"\'')
            
            # Execute tool
            if tool_name == "search":
                results = self.search(tool_args)
                observation = "Observation: Found:\n" + "\n".join(results[:3])
            elif tool_name in self.tools:
                observation = f"Observation: {self.tools[tool_name](tool_args)}"
            else:
                observation = f"Observation: Unknown tool {tool_name}"
            
            conversation_history.append(observation)
        
        return "Could not find answer within iteration limit."

# Usage
def search_kb(query: str) -> List[str]:
    """Search knowledge base (simplified)."""
    results = qdrant_client.search(query, limit=5)
    return [r.payload['text'] for r in results]

tools = {
    "calculator": lambda expr: eval(expr),  # Be careful with eval in production!
    "get_current_date": lambda _: "2024-01-15"
}

agent = AgenticRAG(llm_client, search_kb, tools)

# The agent will decide whether to search or not
answer = agent.query("What is 50% of the RAG system's latency target?")
# Agent thinks:
# Thought: I need to search for the latency target
# Action: search("RAG system latency target")
# Observation: Found: "Target latency is 2 seconds"
# Thought: Now I need to calculate 50% of 2 seconds
# Action: calculator("2 * 0.5")
# Observation: 1.0
# Final Answer: 50% of the RAG system's latency target (2 seconds) is 1 second.
```

**Function Calling (OpenAI/Anthropic Style):**

```python
from openai import OpenAI

class FunctionCallingRAG:
    """RAG using OpenAI function calling."""
    
    def __init__(self, openai_client: OpenAI, qdrant_client):
        self.openai = openai_client
        self.qdrant = qdrant_client
    
    def search_documents(self, query: str) -> str:
        """Search knowledge base - exposed as function."""
        results = self.qdrant.search(query, limit=5)
        return "\n".join([r.payload['text'] for r in results])
    
    def query(self, question: str) -> str:
        """Query with function calling."""
        
        # Define tools/functions
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_documents",
                    "description": "Search the knowledge base for relevant information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
        
        messages = [{"role": "user", "content": question}]
        
        # First call - LLM decides whether to search
        response = self.openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        response_message = response.choices[0].message
        
        # Check if LLM wants to call function
        if response_message.tool_calls:
            # Execute search
            tool_call = response_message.tool_calls[0]
            function_args = json.loads(tool_call.function.arguments)
            search_results = self.search_documents(function_args["query"])
            
            # Add function result to conversation
            messages.append(response_message)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": search_results
            })
            
            # Second call - LLM generates answer with context
            final_response = self.openai.chat.completions.create(
                model="gpt-4-turbo",
                messages=messages
            )
            
            return final_response.choices[0].message.content
        
        # No search needed
        return response_message.content
```

---

### 10.5.2 Self-RAG (Adaptive Retrieval)

**Concept:** The system decides whether retrieval is needed, retrieves if necessary, and self-critiques the generated answer.

**Implementation:**

```python
class SelfRAG:
    """Self-reflective RAG that adapts retrieval strategy."""
    
    def __init__(self, llm_client, qdrant_client, embedding_model):
        self.llm = llm_client
        self.qdrant = qdrant_client
        self.model = embedding_model
    
    def needs_retrieval(self, question: str) -> bool:
        """Decide if question requires external knowledge."""
        
        prompt = f"""Does this question require looking up external information or can it be answered from general knowledge?

Question: {question}

Answer only 'NEEDS_RETRIEVAL' or 'NO_RETRIEVAL'."""
        
        response = self.llm.generate(prompt, max_tokens=10)
        return "NEEDS_RETRIEVAL" in response.upper()
    
    def is_answer_supported(self, answer: str, context: str) -> tuple[bool, float]:
        """Check if answer is supported by context."""
        
        prompt = f"""Is this answer fully supported by the context?

Context: {context}

Answer: {answer}

Rate support from 0.0 (not supported) to 1.0 (fully supported).
Respond with just a number."""
        
        response = self.llm.generate(prompt, max_tokens=10)
        try:
            support_score = float(response.strip())
            is_supported = support_score >= 0.7
            return is_supported, support_score
        except:
            return False, 0.0
    
    def is_answer_useful(self, question: str, answer: str) -> tuple[bool, float]:
        """Check if answer actually addresses the question."""
        
        prompt = f"""Does this answer properly address the question?

Question: {question}

Answer: {answer}

Rate usefulness from 0.0 (not useful) to 1.0 (very useful).
Respond with just a number."""
        
        response = self.llm.generate(prompt, max_tokens=10)
        try:
            usefulness_score = float(response.strip())
            is_useful = usefulness_score >= 0.7
            return is_useful, usefulness_score
        except:
            return False, 0.0
    
    def query(self, question: str) -> Dict:
        """Self-RAG query with adaptive retrieval."""
        
        # Step 1: Decide if retrieval is needed
        needs_docs = self.needs_retrieval(question)
        
        if not needs_docs:
            # Answer from parametric knowledge
            answer = self.llm.generate(
                f"Answer this question: {question}",
                max_tokens=200
            )
            return {
                "answer": answer,
                "retrieval_used": False,
                "confidence": "high"
            }
        
        # Step 2: Retrieve documents
        query_vector = self.model.encode(question).tolist()
        results = self.qdrant.search(query_vector, limit=5)
        context = "\n\n".join([r.payload['text'] for r in results])
        
        # Step 3: Generate answer with context
        prompt = f"""Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"""
        answer = self.llm.generate(prompt, max_tokens=300)
        
        # Step 4: Self-critique
        is_supported, support_score = self.is_answer_supported(answer, context)
        is_useful, usefulness_score = self.is_answer_useful(question, answer)
        
        # Step 5: Decide on final answer
        if is_supported and is_useful:
            return {
                "answer": answer,
                "retrieval_used": True,
                "support_score": support_score,
                "usefulness_score": usefulness_score,
                "confidence": "high"
            }
        elif is_useful but not is_supported:
            # Answer is useful but not grounded - add disclaimer
            return {
                "answer": f"{answer}\n\n⚠️ Note: This answer may not be fully supported by the retrieved documents.",
                "retrieval_used": True,
                "support_score": support_score,
                "usefulness_score": usefulness_score,
                "confidence": "medium"
            }
        else:
            # Answer not good - try parallel retrieval or admit uncertainty
            return {
                "answer": "I don't have sufficient information to answer this question confidently based on the available documents.",
                "retrieval_used": True,
                "support_score": support_score,
                "usefulness_score": usefulness_score,
                "confidence": "low"
            }

# Usage
self_rag = SelfRAG(llm_client, qdrant_client, embedding_model)
result = self_rag.query("What is the capital of France?")
# Result: No retrieval needed (general knowledge)

result = self_rag.query("What is our company's RAG latency target?")
# Result: Retrieves documents, generates answer, critiques itself
```

---

### 10.5.3 Corrective RAG (CRAG)

**Concept:** Detect when retrieval fails and take corrective action (web search, different query, etc.).

```python
from typing import List, Optional
import requests

class CorrectiveRAG:
    """RAG with fallback strategies when retrieval fails."""
    
    def __init__(self, llm_client, qdrant_client, embedding_model, web_search_api_key: str):
        self.llm = llm_client
        self.qdrant = qdrant_client
        self.model = embedding_model
        self.web_search_key = web_search_api_key
    
    def assess_retrieval_quality(self, question: str, retrieved_docs: List[str]) -> float:
        """Assess if retrieved documents are relevant to the question."""
        
        if not retrieved_docs:
            return 0.0
        
        # Use LLM to judge relevance
        docs_sample = "\n".join(retrieved_docs[:3])
        prompt = f"""Rate how relevant these documents are to the question on a scale of 0.0 to 1.0.

Question: {question}

Documents:
{docs_sample}

Relevance score (0.0-1.0):"""
        
        response = self.llm.generate(prompt, max_tokens=10)
        try:
            return float(response.strip())
        except:
            return 0.5  # Default to medium relevance
    
    def web_search(self, query: str) -> List[str]:
        """Fallback to web search (e.g., Serper API, Brave Search)."""
        
        # Example using Serper API
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": self.web_search_key,
            "Content-Type": "application/json"
        }
        data = {"q": query, "num": 5}
        
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            results = response.json().get("organic", [])
            return [f"{r['title']}: {r['snippet']}" for r in results]
        return []
    
    def decompose_query(self, question: str) -> List[str]:
        """Break complex question into sub-questions."""
        
        prompt = f"""Break this complex question into 2-3 simpler sub-questions.

Question: {question}

Sub-questions (one per line):"""
        
        response = self.llm.generate(prompt, max_tokens=150)
        sub_questions = [q.strip() for q in response.strip().split('\n') if q.strip()]
        return sub_questions[:3]
    
    def query(self, question: str) -> Dict:
        """Corrective RAG with multiple fallback strategies."""
        
        # Strategy 1: Try normal RAG retrieval
        query_vector = self.model.encode(question).tolist()
        results = self.qdrant.search(query_vector, limit=10)
        retrieved_docs = [r.payload['text'] for r in results]
        
        # Assess retrieval quality
        quality_score = self.assess_retrieval_quality(question, retrieved_docs)
        
        # Strategy 2: If quality low, try query decomposition
        if quality_score < 0.5:
            print("⚠️ Low retrieval quality. Trying query decomposition...")
            sub_questions = self.decompose_query(question)
            
            all_docs = []
            for sub_q in sub_questions:
                sub_vector = self.model.encode(sub_q).tolist()
                sub_results = self.qdrant.search(sub_vector, limit=5)
                all_docs.extend([r.payload['text'] for r in sub_results])
            
            # Deduplicate
            retrieved_docs = list(set(all_docs))
            quality_score = self.assess_retrieval_quality(question, retrieved_docs)
        
        # Strategy 3: If still poor, fall back to web search
        if quality_score < 0.4:
            print("⚠️ Still low quality. Falling back to web search...")
            web_results = self.web_search(question)
            
            if web_results:
                retrieved_docs = web_results
                quality_score = 0.7  # Assume web is decent quality
        
        # Strategy 4: If all else fails, admit uncertainty
        if quality_score < 0.3:
            return {
                "answer": "I don't have sufficient information to answer this question reliably. The available documents and web search did not provide relevant information.",
                "strategy_used": "none",
                "confidence": "very_low"
            }
        
        # Generate answer with best available context
        context = "\n\n".join(retrieved_docs[:5])
        prompt = f"""Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"""
        answer = self.llm.generate(prompt, max_tokens=400)
        
        strategy = "web_search" if quality_score == 0.7 else "decomposition" if len(retrieved_docs) > 10 else "standard"
        
        return {
            "answer": answer,
            "strategy_used": strategy,
            "quality_score": quality_score,
            "confidence": "high" if quality_score > 0.7 else "medium"
        }

# Usage
crag = CorrectiveRAG(llm_client, qdrant_client, embedding_model, web_search_key)
result = crag.query("What happened in the news today?")
# Falls back to web search since internal docs won't have today's news
```

---

### 10.5.4 Graph RAG (Knowledge Graph Integration)

**Concept:** Combine vector search with knowledge graph traversal for better relational understanding.

```python
from neo4j import GraphDatabase
from typing import List, Tuple

class GraphRAG:
    """RAG with knowledge graph for entity relationships."""
    
    def __init__(self, llm_client, qdrant_client, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        self.llm = llm_client
        self.qdrant = qdrant_client
        self.graph_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    
    def extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text using LLM."""
        
        prompt = f"""Extract the main entities (people, organizations, technologies) from this text.

Text: {text}

Entities (comma-separated):"""
        
        response = self.llm.generate(prompt, max_tokens=100)
        entities = [e.strip() for e in response.split(',')]
        return entities[:5]  # Limit to 5 entities
    
    def get_entity_relationships(self, entity: str, depth: int = 2) -> List[Tuple[str, str, str]]:
        """Get relationships from knowledge graph."""
        
        query = """
        MATCH path = (e:Entity {name: $entity})-[r*1..depth]-(related)
        RETURN e.name as source, type(r[0]) as relationship, related.name as target
        LIMIT 20
        """
        
        with self.graph_driver.session() as session:
            result = session.run(query, entity=entity, depth=depth)
            relationships = [(record["source"], record["relationship"], record["target"]) 
                           for record in result]
        
        return relationships
    
    def expand_context_with_graph(self, chunks: List[str], question: str) -> str:
        """Expand retrieved chunks with graph knowledge."""
        
        # Extract entities from question
        entities = self.extract_entities(question)
        
        # Get relationships for each entity
        graph_context = []
        for entity in entities:
            rels = self.get_entity_relationships(entity, depth=2)
            if rels:
                entity_info = f"\n{entity} relationships:"
                for source, rel, target in rels[:5]:
                    entity_info += f"\n  - {source} {rel} {target}"
                graph_context.append(entity_info)
        
        # Combine vector-retrieved chunks with graph context
        combined_context = "Retrieved Documents:\n" + "\n\n".join(chunks)
        
        if graph_context:
            combined_context += "\n\nKnowledge Graph Information:" + "".join(graph_context)
        
        return combined_context
    
    def query(self, question: str) -> Dict:
        """Query with vector search + knowledge graph."""
        
        # Step 1: Vector search
        query_vector = self.model.encode(question).tolist()
        results = self.qdrant.search(query_vector, limit=5)
        chunks = [r.payload['text'] for r in results]
        
        # Step 2: Expand with knowledge graph
        enhanced_context = self.expand_context_with_graph(chunks, question)
        
        # Step 3: Generate answer
        prompt = f"""Use both the documents and knowledge graph to answer.

{enhanced_context}

Question: {question}

Answer:"""
        
        answer = self.llm.generate(prompt, max_tokens=500)
        
        return {
            "answer": answer,
            "used_graph": len(self.extract_entities(question)) > 0,
            "sources": {"vector_chunks": len(chunks), "graph_entities": len(self.extract_entities(question))}
        }

# Setup: First populate knowledge graph
def build_knowledge_graph_from_documents(documents: List[str], graph_driver):
    """Extract entities and relationships from documents to build graph."""
    
    for doc in documents:
        # Use LLM to extract triples (subject, relation, object)
        prompt = f"""Extract knowledge triples from this text.
        
Text: {doc}

Format: subject | relation | object
One per line."""
        
        response = llm.generate(prompt)
        triples = [line.split('|') for line in response.strip().split('\n')]
        
        # Insert into Neo4j
        with graph_driver.session() as session:
            for triple in triples:
                if len(triple) == 3:
                    subject, relation, obj = [t.strip() for t in triple]
                    query = """
                    MERGE (s:Entity {name: $subject})
                    MERGE (o:Entity {name: $object})
                   MERGE (s)-[r:RELATED {type: $relation}]->(o)
                    """
                    session.run(query, subject=subject, relation=relation, object=obj)

# Usage
graph_rag = GraphRAG(llm_client, qdrant_client, "bolt://localhost:7687", "neo4j", "password")
result = graph_rag.query("How is FastAPI related to Pydantic?")
# Combines vector search with graph traversal to find relationships
```

---

### 10.5.5 Multi-Modal RAG

**Concept:** Handle images, tables, charts, and other non-text content in RAG.

```python
from PIL import Image
import pytesseract
import pandas as pd
from typing import Union
import base64

class MultiModalRAG:
    """RAG that handles text, images, tables, and charts."""
    
    def __init__(self, llm_client, qdrant_client, vision_model):
        self.llm = llm_client
        self.qdrant = qdrant_client
        self.vision_model = vision_model  # e.g., GPT-4 Vision, Claude 3
    
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from images using OCR."""
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text
    
    def describe_image(self, image_path: str) -> str:
        """Get semantic description of image using vision model."""
        
        # Encode image to base64
        with open(image_path, "rb") as img_file:
            image_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Use vision model (GPT-4 Vision example)
        prompt = "Describe this image in detail, including any text, charts, or diagrams."
        
        response = self.vision_model.generate(
            prompt=prompt,
            image_base64=image_data,
            max_tokens=300
        )
        
        return response
    
    def extract_table_data(self, table_html: str) -> str:
        """Convert HTML table to searchable text."""
        
        # Parse table
        df = pd.read_html(table_html)[0]
        
        # Convert to natural language
        table_description = f"Table with {len(df)} rows and {len(df.columns)} columns.\n"
        table_description += f"Columns: {', '.join(df.columns)}\n\n"
        
        # Add sample rows
        for idx, row in df.head(5).iterrows():
            row_text = ", ".join([f"{col}: {val}" for col, val in row.items()])
            table_description += f"Row {idx}: {row_text}\n"
        
        return table_description
    
    def ingest_multimodal_document(self, file_path: str, file_type: str) -> List[Dict]:
        """Ingest document with mixed content types."""
        
        chunks = []
        
        if file_type == "pdf_with_images":
            # Extract text chunks
            text = extract_text_from_pdf(file_path)
            text_chunks = split_into_chunks(text)
            
            # Extract images
            images = extract_images_from_pdf(file_path)
            
            for i, chunk in enumerate(text_chunks):
                chunks.append({
                    "type": "text",
                    "content": chunk,
                    "metadata": {"page": i, "modality": "text"}
                })
            
            for i, img_path in enumerate(images):
                # OCR text
                ocr_text = self.extract_text_from_image(img_path)
                
                # Vision description
                description = self.describe_image(img_path)
                
                # Combined representation
                image_content = f"[IMAGE]\nOCR Text: {ocr_text}\nDescription: {description}"
                
                chunks.append({
                    "type": "image",
                    "content": image_content,
                    "metadata": {"image_path": img_path, "modality": "image"}
                })
        
        return chunks
    
    def query(self, question: str, include_images: bool = True) -> Dict:
        """Multi-modal RAG query."""
        
        # Embed question
        query_vector = self.model.encode(question).tolist()
        
        # Search across all modalities
        results = self.qdrant.search(query_vector, limit=10)
        
        # Separate by modality
        text_chunks = []
        image_chunks = []
        
        for r in results:
            if r.payload.get('modality') == 'text':
                text_chunks.append(r.payload['content'])
            elif r.payload.get('modality') == 'image' and include_images:
                image_chunks.append(r.payload['content'])
        
        # Build multimodal context
        context = "Text Sources:\n" + "\n\n".join(text_chunks[:5])
        
        if image_chunks:
            context += "\n\nImage Sources:\n" + "\n\n".join(image_chunks[:3])
        
        # Generate answer
        prompt = f"""Answer using both text and image information.

{context}

Question: {question}

Answer:"""
        
        answer = self.llm.generate(prompt, max_tokens=500)
        
        return {
            "answer": answer,
            "sources": {
                "text_chunks": len(text_chunks),
                "image_chunks": len(image_chunks)
            }
        }

# Usage
multimodal_rag = MultiModalRAG(llm_client, qdrant_client, vision_model)

# Ingest PDF with charts
chunks = multimodal_rag.ingest_multimodal_document("financial_report.pdf", "pdf_with_images")
# Stores both text and image descriptions in Qdrant

# Query can retrieve both text and images
result = multimodal_rag.query("What does the revenue chart show for Q4?")
# Answer: "According to the chart (from image on page 5), Q4 revenue was $2.5M, 
# a 15% increase from Q3..."
```

---

### 10.5.6 Comparison of Advanced Techniques

| Technique | Best For | Complexity | When to Use |
|-----------|----------|------------|-------------|
| **Agentic RAG** | Complex multi-step queries | High | Questions requiring reasoning + tool use |
| **Self-RAG** | Variable question complexity | Medium | When some questions don't need retrieval |
| **Corrective RAG** | Handling retrieval failures | Medium | When internal KB may be incomplete |
| **Graph RAG** | Relational/entity queries | High | Questions about relationships, hierarchies |
| **Multi-Modal** | Documents with images/charts | High | PDFs, presentations, reports with visuals |

---

### 10.5.7 Hybrid Approach Example

```python
class AdvancedRAG:
    """Combines multiple advanced techniques."""
    
    def __init__(self, components: Dict):
        self.self_rag = components['self_rag']
        self.crag = components['crag']
        self.graph_rag = components.get('graph_rag')
    
    def query(self, question: str) -> Dict:
        """Adaptive RAG that chooses the best strategy."""
        
        # Step 1: Self-RAG decides if retrieval needed
        needs_retrieval = self.self_rag.needs_retrieval(question)
        
        if not needs_retrieval:
            return self.self_rag.query(question)
        
        # Step 2: Try Corrective RAG with fallbacks
        result = self.crag.query(question)
        
        # Step 3: If question involves entities and graph available, use Graph RAG
        if self.graph_rag and self._has_entities(question):
            graph_result = self.graph_rag.query(question)
            
            # Combine results
            result['answer'] += f"\n\nAdditional context from knowledge graph: {graph_result['answer']}"
        
        return result
    
    def _has_entities(self, question: str) -> bool:
        """Check if question involves named entities."""
        # Simple heuristic: check for capitalized words
        words = question.split()
        capitalized = sum(1 for w in words if w[0].isupper())
        return capitalized >= 2

# Usage: Automatically adapts to question type
advanced_rag = AdvancedRAG({
    'self_rag': SelfRAG(...),
    'crag': CorrectiveRAG(...),
    'graph_rag': GraphRAG(...)
})

result = advanced_rag.query("How does our system handle errors?")
# Uses Self-RAG → determines retrieval needed → CRAG retrieves docs

result = advanced_rag.query("What is the relationship between microservices and containers?")
# Uses Graph RAG for entity relationships
```

---

### Resources for Advanced RAG

**Papers:**
- Self-RAG: [https://arxiv.org/abs/2310.11511](https://arxiv.org/abs/2310.11511)
- Corrective RAG (CRAG): [https://arxiv.org/abs/2401.15884](https://arxiv.org/abs/2401.15884)
- Graph RAG: [Microsoft Research](https://www.microsoft.com/en-us/research/blog/graphrag/)
- ReAct: [https://arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629)

**Frameworks:**
- LangChain: Agents with tools
- LlamaIndex: Multi-modal RAG
- Neo4j + LLM: Graph RAG
- Hayden: Multi-modal pipelines

---

