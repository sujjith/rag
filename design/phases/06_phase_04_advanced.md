# Phase 4: Advanced RAG Techniques

> **Goal**: Implement cutting-edge RAG techniques including Agentic RAG, Graph RAG, hybrid search, multi-modal support, and a plugin architecture for extensibility.

---

## Overview

This phase adds advanced capabilities that differentiate your RAG system. By the end, you'll have:

1. Agentic RAG with tool orchestration
2. Graph RAG with knowledge graph integration
3. Multi-modal support (images, tables)
4. Hybrid search (vector + keyword)
5. Advanced retrieval strategies (HyDE, query decomposition)
6. Conversation memory with context management
7. Plugin architecture for extensibility
8. Performance optimization and profiling
9. A/B testing framework for experiments

---

## Technology Stack (Phase 4)

| Component | Choice | Why |
|-----------|--------|-----|
| **Agent Framework** | Custom + LangGraph | Flexible orchestration |
| **Graph Database** | Neo4j | Mature, Cypher query language |
| **Keyword Search** | Elasticsearch / BM25 | Full-text search, hybrid |
| **Image Processing** | CLIP / BLIP-2 | Multi-modal embeddings |
| **Table Extraction** | Camelot / pdfplumber | Structured data from PDFs |
| **Conversation Memory** | Redis + PostgreSQL | Short-term + long-term |
| **Profiling** | py-spy, memory_profiler | Performance analysis |
| **Experiments** | Custom A/B framework | Measure improvements |

---

## Architecture (Phase 4)

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                     Phase 4: Advanced RAG Architecture                            │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   User Query                                                                     │
│       │                                                                           │
│       ▼                                                                           │
│   ┌─────────────────────────────────────────────────────────────────────────┐    │
│   │                        Query Understanding                               │    │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────────┐  │    │
│   │  │   Intent    │  │   Query     │  │  Conversation                   │  │    │
│   │  │   Routing   │  │   Rewrite   │  │  Context                        │  │    │
│   │  └─────────────┘  └─────────────┘  └─────────────────────────────────┘  │    │
│   └─────────────────────────────────────────────────────────────────────────┘    │
│       │                                                                           │
│       ▼                                                                           │
│   ┌─────────────────────────────────────────────────────────────────────────┐    │
│   │                         Agentic RAG Layer                                │    │
│   │  ┌─────────────────────────────────────────────────────────────────┐    │    │
│   │  │                    Agent Orchestrator                            │    │    │
│   │  │   Plan → Execute → Observe → Reflect → Answer                   │    │    │
│   │  └─────────────────────────────────────────────────────────────────┘    │    │
│   │       │              │              │              │                     │    │
│   │       ▼              ▼              ▼              ▼                     │    │
│   │  ┌─────────┐  ┌─────────────┐  ┌─────────┐  ┌─────────────────┐        │    │
│   │  │ Vector  │  │  Knowledge  │  │  Web    │  │  Calculator/    │        │    │
│   │  │ Search  │  │  Graph      │  │  Search │  │  Code Exec      │        │    │
│   │  └─────────┘  └─────────────┘  └─────────┘  └─────────────────┘        │    │
│   └─────────────────────────────────────────────────────────────────────────┘    │
│       │                                                                           │
│       ▼                                                                           │
│   ┌─────────────────────────────────────────────────────────────────────────┐    │
│   │                         Hybrid Retrieval                                 │    │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────────┐  │    │
│   │  │   Vector    │  │  Keyword    │  │  Graph                          │  │    │
│   │  │   (Qdrant)  │  │  (BM25/ES)  │  │  (Neo4j)                        │  │    │
│   │  └──────┬──────┘  └──────┬──────┘  └───────────────┬─────────────────┘  │    │
│   │         │                │                         │                     │    │
│   │         └────────────────┼─────────────────────────┘                     │    │
│   │                          ▼                                               │    │
│   │                   ┌─────────────┐                                        │    │
│   │                   │  Reciprocal │                                        │    │
│   │                   │  Rank Fusion│                                        │    │
│   │                   └─────────────┘                                        │    │
│   └─────────────────────────────────────────────────────────────────────────┘    │
│       │                                                                           │
│       ▼                                                                           │
│   ┌─────────────────────────────────────────────────────────────────────────┐    │
│   │                      Multi-Modal Processing                              │    │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────────┐  │    │
│   │  │    Text     │  │   Images    │  │   Tables                        │  │    │
│   │  │   Chunks    │  │   (CLIP)    │  │   (Structured)                  │  │    │
│   │  └─────────────┘  └─────────────┘  └─────────────────────────────────┘  │    │
│   └─────────────────────────────────────────────────────────────────────────┘    │
│       │                                                                           │
│       ▼                                                                           │
│   ┌─────────────────────────────────────────────────────────────────────────┐    │
│   │                        Response Generation                               │    │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────────┐  │    │
│   │  │  Context    │  │   LLM       │  │  Citation                       │  │    │
│   │  │  Assembly   │  │   Generate  │  │  Formatting                     │  │    │
│   │  └─────────────┘  └─────────────┘  └─────────────────────────────────┘  │    │
│   └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Python Concepts You'll Learn

| Concept | Where Used |
|---------|------------|
| Metaclasses and descriptors | Plugin system |
| Protocol classes (structural typing) | Tool interfaces |
| Custom decorators with arguments | A/B testing, profiling |
| Concurrent.futures | Parallel retrieval |
| Context variables | Conversation state |
| Generator expressions | Streaming responses |
| Memory profiling | Optimization |
| cProfile and py-spy | Performance analysis |
| Abstract factory pattern | Retriever creation |
| Observer pattern | Event-driven agents |

---

## Project Structure

```
rag_v1/
├── src/
│   └── rag/
│       ├── __init__.py
│       ├── config.py
│       │
│       ├── agents/                    # NEW: Agentic RAG
│       │   ├── __init__.py
│       │   ├── base.py                # Agent base classes
│       │   ├── orchestrator.py        # Agent orchestration
│       │   ├── tools/                 # Agent tools
│       │   │   ├── __init__.py
│       │   │   ├── base.py            # Tool protocol
│       │   │   ├── vector_search.py   # Vector search tool
│       │   │   ├── graph_search.py    # Graph search tool
│       │   │   ├── web_search.py      # Web search tool
│       │   │   ├── calculator.py      # Math tool
│       │   │   └── code_executor.py   # Code execution tool
│       │   └── prompts/
│       │       ├── planner.py
│       │       └── executor.py
│       │
│       ├── graph/                     # NEW: Graph RAG
│       │   ├── __init__.py
│       │   ├── neo4j_client.py        # Neo4j connection
│       │   ├── entity_extractor.py    # NER for graph building
│       │   ├── relationship_extractor.py
│       │   ├── graph_builder.py       # Build knowledge graph
│       │   └── graph_retriever.py     # Query graph
│       │
│       ├── hybrid/                    # NEW: Hybrid Search
│       │   ├── __init__.py
│       │   ├── bm25.py                # BM25 implementation
│       │   ├── elasticsearch.py       # ES integration
│       │   ├── fusion.py              # Reciprocal Rank Fusion
│       │   └── hybrid_retriever.py    # Combined retrieval
│       │
│       ├── multimodal/                # NEW: Multi-modal
│       │   ├── __init__.py
│       │   ├── image_embedder.py      # CLIP embeddings
│       │   ├── image_captioner.py     # Image descriptions
│       │   ├── table_extractor.py     # Extract tables from PDFs
│       │   └── multimodal_chunker.py  # Handle mixed content
│       │
│       ├── conversation/              # NEW: Conversation Memory
│       │   ├── __init__.py
│       │   ├── memory.py              # Memory management
│       │   ├── context_builder.py     # Build conversation context
│       │   └── session.py             # Session management
│       │
│       ├── strategies/                # NEW: Advanced Retrieval
│       │   ├── __init__.py
│       │   ├── hyde.py                # Hypothetical Document Embeddings
│       │   ├── decomposition.py       # Query decomposition
│       │   ├── step_back.py           # Step-back prompting
│       │   └── self_query.py          # Self-querying retrieval
│       │
│       ├── plugins/                   # NEW: Plugin Architecture
│       │   ├── __init__.py
│       │   ├── base.py                # Plugin protocol
│       │   ├── registry.py            # Plugin registry
│       │   ├── loader.py              # Dynamic loading
│       │   └── hooks.py               # Extension points
│       │
│       ├── experiments/               # NEW: A/B Testing
│       │   ├── __init__.py
│       │   ├── framework.py           # Experiment framework
│       │   ├── variants.py            # Variant management
│       │   ├── metrics.py             # Experiment metrics
│       │   └── analysis.py            # Statistical analysis
│       │
│       ├── profiling/                 # NEW: Performance
│       │   ├── __init__.py
│       │   ├── decorators.py          # Profiling decorators
│       │   ├── memory.py              # Memory profiling
│       │   └── tracing.py             # Request tracing
│       │
│       ├── api/                       # (Updated)
│       │   └── routes/
│       │       ├── agents.py          # Agent endpoints
│       │       ├── graph.py           # Graph endpoints
│       │       └── experiments.py     # Experiment endpoints
│       │
│       └── ... (previous modules)
│
└── plugins/                           # External plugins directory
    └── example_plugin/
        ├── __init__.py
        └── plugin.py
```

---

## Implementation Tasks

| # | Task | Priority | Python Concepts | Files |
|---|------|----------|-----------------|-------|
| 1 | Add new dependencies | High | `pyproject.toml` | `pyproject.toml` |
| 2 | Tool protocol and base | High | Protocol, ABC | `agents/tools/base.py` |
| 3 | Agent orchestrator | High | State machine, async | `agents/orchestrator.py` |
| 4 | Vector search tool | High | Tool implementation | `agents/tools/vector_search.py` |
| 5 | Neo4j integration | High | Graph database | `graph/neo4j_client.py` |
| 6 | Entity extraction | Medium | NER, spaCy | `graph/entity_extractor.py` |
| 7 | Knowledge graph builder | High | Graph algorithms | `graph/graph_builder.py` |
| 8 | BM25 keyword search | High | TF-IDF, ranking | `hybrid/bm25.py` |
| 9 | Reciprocal Rank Fusion | High | Score combination | `hybrid/fusion.py` |
| 10 | CLIP image embeddings | Medium | Multi-modal | `multimodal/image_embedder.py` |
| 11 | Table extraction | Medium | PDF parsing | `multimodal/table_extractor.py` |
| 12 | Conversation memory | High | State management | `conversation/memory.py` |
| 13 | HyDE retrieval | Medium | Query expansion | `strategies/hyde.py` |
| 14 | Query decomposition | Medium | Chain-of-thought | `strategies/decomposition.py` |
| 15 | Plugin architecture | High | Metaclasses, loading | `plugins/` |
| 16 | A/B testing framework | Medium | Statistics | `experiments/` |
| 17 | Performance profiling | Medium | cProfile, memory | `profiling/` |
| 18 | API endpoints | High | FastAPI routes | `api/routes/` |

---

## Step-by-Step Implementation

### Step 1: Add New Dependencies

```bash
# Agent & NLP
uv add spacy tiktoken

# Graph database
uv add neo4j

# Keyword search
uv add rank-bm25 elasticsearch

# Multi-modal
uv add transformers pillow camelot-py[cv] pdfplumber

# Performance
uv add py-spy memory-profiler line-profiler

# Download spaCy model
python -m spacy download en_core_web_sm
```

**Update pyproject.toml:**
```toml
[project]
name = "rag-system"
version = "0.4.0"
description = "Advanced RAG system with agentic capabilities"
requires-python = ">=3.11"

dependencies = [
    # ... Phase 1-3 dependencies ...
    # Phase 4 dependencies
    "spacy>=3.7.0",
    "tiktoken>=0.5.0",
    "neo4j>=5.15.0",
    "rank-bm25>=0.2.2",
    "elasticsearch>=8.12.0",
    "transformers>=4.36.0",
    "pillow>=10.2.0",
    "pdfplumber>=0.10.0",
    "scipy>=1.12.0",
]

[project.optional-dependencies]
multimodal = [
    "camelot-py[cv]>=0.11.0",
    "torch>=2.1.0",
    "open-clip-torch>=2.24.0",
]
profiling = [
    "py-spy>=0.3.14",
    "memory-profiler>=0.61.0",
    "line-profiler>=4.1.0",
]
```

---

### Step 2: Tool Protocol and Base

```python
# src/rag/agents/tools/base.py
"""Base tool definitions using Protocol for structural typing."""
from typing import Protocol, Any, runtime_checkable
from dataclasses import dataclass
from abc import abstractmethod
from enum import Enum


class ToolCategory(Enum):
    """Categories of tools."""
    RETRIEVAL = "retrieval"
    COMPUTATION = "computation"
    EXTERNAL = "external"
    UTILITY = "utility"


@dataclass
class ToolResult:
    """Result from tool execution."""
    success: bool
    data: Any
    error: str | None = None
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ToolSpec:
    """Tool specification for LLM."""
    name: str
    description: str
    parameters: dict  # JSON Schema
    category: ToolCategory
    examples: list[dict] = None

    def to_openai_function(self) -> dict:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }


@runtime_checkable
class Tool(Protocol):
    """Protocol for agent tools."""

    @property
    def spec(self) -> ToolSpec:
        """Get tool specification."""
        ...

    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        ...

    def validate_params(self, **kwargs) -> bool:
        """Validate input parameters."""
        ...


class BaseTool:
    """Base implementation for tools."""

    def __init__(self):
        self._spec: ToolSpec | None = None

    @property
    def spec(self) -> ToolSpec:
        if self._spec is None:
            raise NotImplementedError("Tool must define spec")
        return self._spec

    def validate_params(self, **kwargs) -> bool:
        """Basic parameter validation."""
        required = self.spec.parameters.get("required", [])
        return all(k in kwargs for k in required)

    async def execute(self, **kwargs) -> ToolResult:
        if not self.validate_params(**kwargs):
            return ToolResult(
                success=False,
                data=None,
                error="Invalid parameters"
            )
        return await self._execute(**kwargs)

    async def _execute(self, **kwargs) -> ToolResult:
        """Override this method in subclasses."""
        raise NotImplementedError
```

---

### Step 3: Agent Orchestrator

```python
# src/rag/agents/orchestrator.py
"""Agent orchestrator for multi-step reasoning."""
from typing import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
import json

from rag.agents.tools.base import Tool, ToolResult, ToolSpec
from rag.llm.groq import GroqLLM
from rag.logging_config import get_logger
from rag.config import settings

logger = get_logger("agent")


class AgentState(Enum):
    """Agent execution states."""
    PLANNING = "planning"
    EXECUTING = "executing"
    OBSERVING = "observing"
    REFLECTING = "reflecting"
    ANSWERING = "answering"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class AgentStep:
    """A single step in agent execution."""
    state: AgentState
    thought: str
    action: str | None = None
    action_input: dict | None = None
    observation: str | None = None
    tool_result: ToolResult | None = None


@dataclass
class AgentContext:
    """Context for agent execution."""
    query: str
    conversation_history: list[dict] = field(default_factory=list)
    steps: list[AgentStep] = field(default_factory=list)
    max_steps: int = 10
    current_step: int = 0


class AgentOrchestrator:
    """
    ReAct-style agent orchestrator.

    Implements: Thought → Action → Observation → Repeat → Answer
    """

    SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools.

When answering questions, follow this process:
1. Think about what information you need
2. Use tools to gather information
3. Observe the results
4. Decide if you need more information or can answer
5. Provide a comprehensive answer with citations

Available tools:
{tools}

Always respond in this JSON format:
{{
    "thought": "Your reasoning about what to do next",
    "action": "tool_name or 'answer'",
    "action_input": {{"param": "value"}} or {{"answer": "your final answer"}}
}}

Important:
- Use tools to find information, don't make up facts
- Cite your sources in the final answer
- If tools don't provide enough info, say so honestly
"""

    def __init__(self, tools: list[Tool], llm: GroqLLM = None):
        self.tools = {tool.spec.name: tool for tool in tools}
        self.llm = llm or GroqLLM()
        self.tool_specs = [tool.spec for tool in tools]

    async def run(
        self,
        query: str,
        conversation_history: list[dict] = None,
        max_steps: int = 10,
    ) -> AsyncIterator[AgentStep]:
        """
        Run the agent on a query.

        Yields AgentStep objects for each step in execution.
        """
        context = AgentContext(
            query=query,
            conversation_history=conversation_history or [],
            max_steps=max_steps,
        )

        while context.current_step < context.max_steps:
            context.current_step += 1

            # Plan next action
            step = await self._plan(context)
            yield step

            if step.state == AgentState.ERROR:
                break

            # Check if agent wants to answer
            if step.action == "answer":
                step.state = AgentState.COMPLETE
                yield step
                break

            # Execute tool
            if step.action and step.action in self.tools:
                step.state = AgentState.EXECUTING
                yield step

                result = await self._execute_tool(step.action, step.action_input)
                step.tool_result = result
                step.observation = self._format_observation(result)

                step.state = AgentState.OBSERVING
                yield step

            context.steps.append(step)

        # If we hit max steps without answering
        if context.current_step >= context.max_steps:
            final_step = AgentStep(
                state=AgentState.ERROR,
                thought="Reached maximum steps without finding answer",
                action="answer",
                action_input={"answer": "I couldn't find a complete answer within the allowed steps."}
            )
            yield final_step

    async def _plan(self, context: AgentContext) -> AgentStep:
        """Plan the next action based on context."""
        # Build prompt with tool descriptions
        tools_desc = "\n".join([
            f"- {spec.name}: {spec.description}"
            for spec in self.tool_specs
        ])

        system = self.SYSTEM_PROMPT.format(tools=tools_desc)

        # Build conversation with previous steps
        messages = [{"role": "system", "content": system}]

        # Add conversation history
        for msg in context.conversation_history:
            messages.append(msg)

        # Add current query
        messages.append({"role": "user", "content": context.query})

        # Add previous steps
        for step in context.steps:
            messages.append({
                "role": "assistant",
                "content": json.dumps({
                    "thought": step.thought,
                    "action": step.action,
                    "action_input": step.action_input
                })
            })
            if step.observation:
                messages.append({
                    "role": "user",
                    "content": f"Observation: {step.observation}"
                })

        # Get LLM response
        try:
            response = self.llm.client.chat.completions.create(
                model=settings.groq_model,
                messages=messages,
                temperature=0.1,
                max_tokens=1024,
            )

            content = response.choices[0].message.content

            # Parse JSON response
            parsed = json.loads(content)

            return AgentStep(
                state=AgentState.PLANNING,
                thought=parsed.get("thought", ""),
                action=parsed.get("action"),
                action_input=parsed.get("action_input"),
            )

        except json.JSONDecodeError as e:
            logger.error("agent_parse_error", error=str(e))
            return AgentStep(
                state=AgentState.ERROR,
                thought=f"Failed to parse response: {str(e)}",
            )
        except Exception as e:
            logger.error("agent_error", error=str(e))
            return AgentStep(
                state=AgentState.ERROR,
                thought=f"Error: {str(e)}",
            )

    async def _execute_tool(self, tool_name: str, params: dict) -> ToolResult:
        """Execute a tool."""
        tool = self.tools.get(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                data=None,
                error=f"Unknown tool: {tool_name}"
            )

        try:
            return await tool.execute(**(params or {}))
        except Exception as e:
            logger.error("tool_execution_error", tool=tool_name, error=str(e))
            return ToolResult(
                success=False,
                data=None,
                error=str(e)
            )

    def _format_observation(self, result: ToolResult) -> str:
        """Format tool result as observation."""
        if result.success:
            if isinstance(result.data, list):
                return f"Found {len(result.data)} results:\n" + "\n".join(
                    str(item)[:500] for item in result.data[:5]
                )
            return str(result.data)[:2000]
        return f"Tool error: {result.error}"
```

---

### Step 4: Vector Search Tool

```python
# src/rag/agents/tools/vector_search.py
"""Vector search tool for agents."""
from rag.agents.tools.base import BaseTool, ToolSpec, ToolResult, ToolCategory
from rag.embeddings.embedder import Embedder
from rag.vectorstore.qdrant import QdrantStore
from rag.retrieval.reranker import Reranker
from rag.config import settings


class VectorSearchTool(BaseTool):
    """Tool for semantic vector search."""

    def __init__(
        self,
        embedder: Embedder = None,
        store: QdrantStore = None,
        reranker: Reranker = None,
        collection_name: str = None,
    ):
        super().__init__()
        self.embedder = embedder or Embedder()
        self.store = store or QdrantStore(
            collection_name=collection_name or settings.collection_name
        )
        self.reranker = reranker

        self._spec = ToolSpec(
            name="vector_search",
            description="Search documents using semantic similarity. Use this to find relevant information from the knowledge base.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 5
                    },
                    "rerank": {
                        "type": "boolean",
                        "description": "Whether to rerank results",
                        "default": False
                    }
                },
                "required": ["query"]
            },
            category=ToolCategory.RETRIEVAL,
            examples=[
                {"query": "How does authentication work?", "top_k": 5},
                {"query": "What are the API endpoints?", "top_k": 3, "rerank": True}
            ]
        )

    async def _execute(
        self,
        query: str,
        top_k: int = 5,
        rerank: bool = False,
        **kwargs
    ) -> ToolResult:
        """Execute vector search."""
        try:
            # Embed query
            query_vector = self.embedder.embed_query(query)

            # Search
            fetch_k = top_k * 2 if rerank else top_k
            results = self.store.search(query_vector, top_k=fetch_k)

            # Rerank if enabled
            if rerank and results and self.reranker:
                results = self.reranker.rerank(query, results, top_k=top_k)
            else:
                results = results[:top_k]

            # Format results
            formatted = [
                {
                    "content": chunk.content,
                    "source": chunk.metadata.get("filename", "Unknown"),
                    "score": score,
                    "chunk_id": chunk.id,
                }
                for chunk, score in results
            ]

            return ToolResult(
                success=True,
                data=formatted,
                metadata={"query": query, "results_count": len(formatted)}
            )

        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=str(e)
            )
```

---

### Step 5: Neo4j Integration

```python
# src/rag/graph/neo4j_client.py
"""Neo4j graph database client."""
from contextlib import asynccontextmanager
from typing import Any
from neo4j import AsyncGraphDatabase, AsyncDriver
from rag.config import settings
from rag.logging_config import get_logger

logger = get_logger("neo4j")


class Neo4jClient:
    """Async Neo4j client."""

    def __init__(
        self,
        uri: str = None,
        user: str = None,
        password: str = None,
    ):
        self.uri = uri or getattr(settings, "neo4j_uri", "bolt://localhost:7687")
        self.user = user or getattr(settings, "neo4j_user", "neo4j")
        self.password = password or getattr(settings, "neo4j_password", "password")
        self._driver: AsyncDriver | None = None

    async def connect(self) -> None:
        """Connect to Neo4j."""
        self._driver = AsyncGraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password)
        )
        # Verify connection
        async with self._driver.session() as session:
            await session.run("RETURN 1")
        logger.info("neo4j_connected", uri=self.uri)

    async def close(self) -> None:
        """Close connection."""
        if self._driver:
            await self._driver.close()
            self._driver = None

    @asynccontextmanager
    async def session(self):
        """Get a session context manager."""
        if not self._driver:
            await self.connect()
        async with self._driver.session() as session:
            yield session

    async def execute_query(
        self,
        query: str,
        parameters: dict = None,
    ) -> list[dict]:
        """Execute a Cypher query."""
        async with self.session() as session:
            result = await session.run(query, parameters or {})
            records = await result.data()
            return records

    async def create_node(
        self,
        label: str,
        properties: dict,
    ) -> str:
        """Create a node and return its ID."""
        query = f"""
        CREATE (n:{label} $props)
        RETURN elementId(n) as id
        """
        result = await self.execute_query(query, {"props": properties})
        return result[0]["id"] if result else None

    async def create_relationship(
        self,
        from_id: str,
        to_id: str,
        rel_type: str,
        properties: dict = None,
    ) -> None:
        """Create a relationship between nodes."""
        query = f"""
        MATCH (a), (b)
        WHERE elementId(a) = $from_id AND elementId(b) = $to_id
        CREATE (a)-[r:{rel_type} $props]->(b)
        """
        await self.execute_query(query, {
            "from_id": from_id,
            "to_id": to_id,
            "props": properties or {}
        })

    async def find_related(
        self,
        entity: str,
        relationship: str = None,
        depth: int = 2,
        limit: int = 20,
    ) -> list[dict]:
        """Find entities related to a given entity."""
        rel_pattern = f":{relationship}" if relationship else ""
        query = f"""
        MATCH (e:Entity {{name: $entity}})
        MATCH path = (e)-[{rel_pattern}*1..{depth}]-(related)
        RETURN DISTINCT related.name as name,
               related.type as type,
               [r in relationships(path) | type(r)] as relationships,
               length(path) as distance
        ORDER BY distance
        LIMIT $limit
        """
        return await self.execute_query(query, {
            "entity": entity,
            "limit": limit
        })

    async def search_entities(
        self,
        query: str,
        entity_type: str = None,
        limit: int = 10,
    ) -> list[dict]:
        """Full-text search for entities."""
        type_filter = f"AND e.type = '{entity_type}'" if entity_type else ""
        cypher = f"""
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower($query) {type_filter}
        RETURN e.name as name, e.type as type, e.description as description
        LIMIT $limit
        """
        return await self.execute_query(cypher, {
            "query": query,
            "limit": limit
        })
```

---

### Step 6: Entity Extraction

```python
# src/rag/graph/entity_extractor.py
"""Extract entities from text using spaCy."""
import spacy
from dataclasses import dataclass
from typing import Iterator
from rag.logging_config import get_logger

logger = get_logger("entity_extractor")


@dataclass
class Entity:
    """Extracted entity."""
    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0


@dataclass
class Relationship:
    """Extracted relationship between entities."""
    subject: Entity
    predicate: str
    object: Entity
    confidence: float = 1.0


class EntityExtractor:
    """Extract named entities using spaCy."""

    # Map spaCy labels to our entity types
    LABEL_MAP = {
        "PERSON": "Person",
        "ORG": "Organization",
        "GPE": "Location",
        "LOC": "Location",
        "DATE": "Date",
        "TIME": "Time",
        "MONEY": "Money",
        "PRODUCT": "Product",
        "EVENT": "Event",
        "WORK_OF_ART": "Work",
        "LAW": "Law",
        "LANGUAGE": "Language",
        "NORP": "Group",
        "FAC": "Facility",
    }

    def __init__(self, model: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model)
        except OSError:
            logger.warning("spacy_model_not_found", model=model)
            # Download model
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", model])
            self.nlp = spacy.load(model)

    def extract_entities(self, text: str) -> list[Entity]:
        """Extract named entities from text."""
        doc = self.nlp(text)

        entities = []
        for ent in doc.ents:
            label = self.LABEL_MAP.get(ent.label_, ent.label_)
            entities.append(Entity(
                text=ent.text,
                label=label,
                start=ent.start_char,
                end=ent.end_char,
            ))

        return entities

    def extract_noun_chunks(self, text: str) -> list[str]:
        """Extract noun chunks as potential entities."""
        doc = self.nlp(text)
        return [chunk.text for chunk in doc.noun_chunks]

    def extract_relationships(self, text: str) -> list[Relationship]:
        """
        Extract basic relationships using dependency parsing.

        Note: For production, consider using an LLM for better extraction.
        """
        doc = self.nlp(text)
        relationships = []

        for sent in doc.sents:
            # Find subject-verb-object patterns
            subjects = []
            objects = []
            root = None

            for token in sent:
                if token.dep_ == "ROOT":
                    root = token
                elif token.dep_ in ("nsubj", "nsubjpass"):
                    subjects.append(token)
                elif token.dep_ in ("dobj", "pobj", "attr"):
                    objects.append(token)

            if root and subjects and objects:
                for subj in subjects:
                    for obj in objects:
                        # Get entity info if available
                        subj_ent = self._get_entity_for_token(subj, doc)
                        obj_ent = self._get_entity_for_token(obj, doc)

                        if subj_ent and obj_ent:
                            relationships.append(Relationship(
                                subject=subj_ent,
                                predicate=root.lemma_,
                                object=obj_ent,
                            ))

        return relationships

    def _get_entity_for_token(self, token, doc) -> Entity | None:
        """Get entity for a token if it's part of an entity."""
        for ent in doc.ents:
            if ent.start <= token.i < ent.end:
                return Entity(
                    text=ent.text,
                    label=self.LABEL_MAP.get(ent.label_, ent.label_),
                    start=ent.start_char,
                    end=ent.end_char,
                )
        # Return token as generic entity
        return Entity(
            text=token.text,
            label="Concept",
            start=token.idx,
            end=token.idx + len(token.text),
        )


class LLMEntityExtractor:
    """Extract entities using LLM for better accuracy."""

    EXTRACTION_PROMPT = """Extract all named entities and their relationships from the following text.

Text:
{text}

Return a JSON object with:
{{
    "entities": [
        {{"text": "entity name", "type": "Person|Organization|Location|Concept|Event|Product"}}
    ],
    "relationships": [
        {{"subject": "entity1", "predicate": "relationship", "object": "entity2"}}
    ]
}}

Focus on:
- People, organizations, locations
- Technical concepts and products
- Relationships like "works_for", "located_in", "part_of", "created_by"
"""

    def __init__(self, llm=None):
        from rag.llm.groq import GroqLLM
        self.llm = llm or GroqLLM()

    async def extract(self, text: str) -> tuple[list[Entity], list[Relationship]]:
        """Extract entities and relationships using LLM."""
        import json

        prompt = self.EXTRACTION_PROMPT.format(text=text[:4000])

        response = self.llm.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=2000,
        )

        try:
            content = response.choices[0].message.content
            # Find JSON in response
            start = content.find("{")
            end = content.rfind("}") + 1
            data = json.loads(content[start:end])

            entities = [
                Entity(text=e["text"], label=e["type"], start=0, end=0)
                for e in data.get("entities", [])
            ]

            relationships = []
            for r in data.get("relationships", []):
                subj = Entity(text=r["subject"], label="Entity", start=0, end=0)
                obj = Entity(text=r["object"], label="Entity", start=0, end=0)
                relationships.append(Relationship(
                    subject=subj,
                    predicate=r["predicate"],
                    object=obj,
                ))

            return entities, relationships

        except (json.JSONDecodeError, KeyError) as e:
            logger.error("llm_extraction_error", error=str(e))
            return [], []
```

---

### Step 7: Knowledge Graph Builder

```python
# src/rag/graph/graph_builder.py
"""Build knowledge graph from documents."""
from typing import AsyncIterator
from dataclasses import dataclass

from rag.graph.neo4j_client import Neo4jClient
from rag.graph.entity_extractor import EntityExtractor, LLMEntityExtractor, Entity, Relationship
from rag.models import Document, Chunk
from rag.logging_config import get_logger

logger = get_logger("graph_builder")


@dataclass
class GraphStats:
    """Statistics about graph building."""
    nodes_created: int = 0
    relationships_created: int = 0
    documents_processed: int = 0


class KnowledgeGraphBuilder:
    """Build and maintain a knowledge graph from documents."""

    def __init__(
        self,
        neo4j_client: Neo4jClient = None,
        entity_extractor: EntityExtractor = None,
        use_llm_extraction: bool = False,
    ):
        self.client = neo4j_client or Neo4jClient()
        self.extractor = entity_extractor or EntityExtractor()
        self.llm_extractor = LLMEntityExtractor() if use_llm_extraction else None

    async def initialize(self) -> None:
        """Initialize the graph with constraints and indexes."""
        await self.client.connect()

        # Create constraints
        constraints = [
            "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
        ]

        for constraint in constraints:
            try:
                await self.client.execute_query(constraint)
            except Exception as e:
                logger.debug("constraint_exists", error=str(e))

        # Create indexes
        indexes = [
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX chunk_document IF NOT EXISTS FOR (c:Chunk) ON (c.document_id)",
        ]

        for index in indexes:
            try:
                await self.client.execute_query(index)
            except Exception as e:
                logger.debug("index_exists", error=str(e))

    async def process_document(self, document: Document, chunks: list[Chunk]) -> GraphStats:
        """Process a document and add to knowledge graph."""
        stats = GraphStats()

        # Create document node
        doc_node_id = await self._create_document_node(document)
        stats.documents_processed = 1

        # Process each chunk
        for chunk in chunks:
            chunk_node_id = await self._create_chunk_node(chunk, doc_node_id)

            # Extract entities
            if self.llm_extractor:
                entities, relationships = await self.llm_extractor.extract(chunk.content)
            else:
                entities = self.extractor.extract_entities(chunk.content)
                relationships = self.extractor.extract_relationships(chunk.content)

            # Create entity nodes and link to chunk
            entity_ids = {}
            for entity in entities:
                entity_id = await self._create_or_get_entity(entity)
                entity_ids[entity.text] = entity_id

                # Link entity to chunk
                await self.client.create_relationship(
                    chunk_node_id, entity_id, "MENTIONS"
                )
                stats.nodes_created += 1

            # Create relationships between entities
            for rel in relationships:
                if rel.subject.text in entity_ids and rel.object.text in entity_ids:
                    await self.client.create_relationship(
                        entity_ids[rel.subject.text],
                        entity_ids[rel.object.text],
                        rel.predicate.upper().replace(" ", "_"),
                        {"confidence": rel.confidence}
                    )
                    stats.relationships_created += 1

        logger.info(
            "document_processed",
            doc_id=document.id,
            nodes=stats.nodes_created,
            relationships=stats.relationships_created
        )

        return stats

    async def _create_document_node(self, document: Document) -> str:
        """Create a document node."""
        query = """
        MERGE (d:Document {doc_id: $doc_id})
        ON CREATE SET d.source = $source, d.created_at = datetime()
        ON MATCH SET d.updated_at = datetime()
        RETURN elementId(d) as id
        """
        result = await self.client.execute_query(query, {
            "doc_id": document.id,
            "source": document.source,
        })
        return result[0]["id"]

    async def _create_chunk_node(self, chunk: Chunk, doc_node_id: str) -> str:
        """Create a chunk node and link to document."""
        query = """
        MATCH (d:Document) WHERE elementId(d) = $doc_id
        CREATE (c:Chunk {
            chunk_id: $chunk_id,
            content: $content,
            document_id: $document_id,
            start_char: $start_char,
            end_char: $end_char
        })
        CREATE (c)-[:PART_OF]->(d)
        RETURN elementId(c) as id
        """
        result = await self.client.execute_query(query, {
            "doc_id": doc_node_id,
            "chunk_id": chunk.id,
            "content": chunk.content[:1000],  # Limit content size
            "document_id": chunk.document_id,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
        })
        return result[0]["id"]

    async def _create_or_get_entity(self, entity: Entity) -> str:
        """Create an entity node or get existing."""
        query = """
        MERGE (e:Entity {name: $name})
        ON CREATE SET e.type = $type, e.created_at = datetime()
        ON MATCH SET e.mention_count = COALESCE(e.mention_count, 0) + 1
        RETURN elementId(e) as id
        """
        result = await self.client.execute_query(query, {
            "name": entity.text,
            "type": entity.label,
        })
        return result[0]["id"]

    async def query_related_context(
        self,
        entities: list[str],
        depth: int = 2,
        limit: int = 10,
    ) -> list[dict]:
        """Query graph for context related to entities."""
        if not entities:
            return []

        query = """
        UNWIND $entities as entity_name
        MATCH (e:Entity {name: entity_name})
        MATCH path = (e)-[*1..2]-(related:Entity)
        WITH DISTINCT related, e,
             [r in relationships(path) | type(r)] as rels
        MATCH (related)<-[:MENTIONS]-(c:Chunk)
        RETURN related.name as entity,
               related.type as type,
               collect(DISTINCT c.content)[0..3] as contexts,
               rels as relationships
        LIMIT $limit
        """
        return await self.client.execute_query(query, {
            "entities": entities,
            "limit": limit,
        })
```

---

### Step 8: BM25 Keyword Search

```python
# src/rag/hybrid/bm25.py
"""BM25 keyword search implementation."""
import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterator
import re

from rag.models import Chunk
from rag.logging_config import get_logger

logger = get_logger("bm25")


@dataclass
class BM25Result:
    """BM25 search result."""
    chunk: Chunk
    score: float
    matched_terms: list[str]


class BM25Index:
    """
    BM25 (Best Matching 25) implementation for keyword search.

    BM25 is a bag-of-words retrieval function that ranks documents
    based on query terms appearing in each document.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ):
        """
        Initialize BM25.

        Args:
            k1: Term frequency saturation parameter (1.2-2.0)
            b: Length normalization parameter (0-1)
            epsilon: Floor for IDF to handle zero IDF
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        self.corpus: list[Chunk] = []
        self.doc_freqs: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.doc_len: list[int] = []
        self.avgdl: float = 0
        self.tokenized_corpus: list[list[str]] = []

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization."""
        # Lowercase, remove punctuation, split on whitespace
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        # Remove stopwords (basic)
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was',
                     'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
                     'does', 'did', 'will', 'would', 'could', 'should', 'may',
                     'might', 'must', 'shall', 'can', 'to', 'of', 'in', 'for',
                     'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
                     'during', 'before', 'after', 'above', 'below', 'between',
                     'under', 'again', 'further', 'then', 'once', 'here', 'there',
                     'when', 'where', 'why', 'how', 'all', 'each', 'few', 'more',
                     'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
                     'own', 'same', 'so', 'than', 'too', 'very', 'just', 'it', 'its'}
        return [t for t in tokens if t not in stopwords and len(t) > 1]

    def index(self, chunks: list[Chunk]) -> None:
        """Build BM25 index from chunks."""
        self.corpus = chunks
        self.tokenized_corpus = []
        self.doc_len = []
        self.doc_freqs = Counter()

        # Tokenize and count
        for chunk in chunks:
            tokens = self._tokenize(chunk.content)
            self.tokenized_corpus.append(tokens)
            self.doc_len.append(len(tokens))

            # Count document frequency (unique terms per doc)
            unique_terms = set(tokens)
            for term in unique_terms:
                self.doc_freqs[term] += 1

        # Calculate average document length
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0

        # Calculate IDF
        n_docs = len(chunks)
        for term, freq in self.doc_freqs.items():
            # IDF with smoothing
            idf = math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1)
            self.idf[term] = max(idf, self.epsilon)

        logger.info("bm25_index_built", docs=len(chunks), terms=len(self.idf))

    def search(self, query: str, top_k: int = 10) -> list[BM25Result]:
        """Search the index with a query."""
        if not self.corpus:
            return []

        query_tokens = self._tokenize(query)
        scores = []

        for idx, doc_tokens in enumerate(self.tokenized_corpus):
            score = self._score_document(query_tokens, doc_tokens, idx)
            if score > 0:
                matched = [t for t in query_tokens if t in doc_tokens]
                scores.append((idx, score, matched))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score, matched in scores[:top_k]:
            results.append(BM25Result(
                chunk=self.corpus[idx],
                score=score,
                matched_terms=matched
            ))

        return results

    def _score_document(
        self,
        query_tokens: list[str],
        doc_tokens: list[str],
        doc_idx: int,
    ) -> float:
        """Calculate BM25 score for a document."""
        score = 0.0
        doc_len = self.doc_len[doc_idx]
        term_freqs = Counter(doc_tokens)

        for term in query_tokens:
            if term not in self.idf:
                continue

            tf = term_freqs.get(term, 0)
            if tf == 0:
                continue

            idf = self.idf[term]

            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * numerator / denominator

        return score

    def add_document(self, chunk: Chunk) -> None:
        """Add a single document to the index."""
        tokens = self._tokenize(chunk.content)

        self.corpus.append(chunk)
        self.tokenized_corpus.append(tokens)
        self.doc_len.append(len(tokens))

        # Update stats
        n_docs = len(self.corpus)
        self.avgdl = sum(self.doc_len) / n_docs

        # Update document frequencies and IDF
        unique_terms = set(tokens)
        for term in unique_terms:
            self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1
            freq = self.doc_freqs[term]
            idf = math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1)
            self.idf[term] = max(idf, self.epsilon)
```

---

### Step 9: Reciprocal Rank Fusion

```python
# src/rag/hybrid/fusion.py
"""Result fusion strategies for hybrid search."""
from dataclasses import dataclass
from typing import TypeVar, Generic
from collections import defaultdict

T = TypeVar("T")


@dataclass
class FusedResult(Generic[T]):
    """Result after fusion."""
    item: T
    score: float
    sources: dict[str, float]  # source_name -> original_score


class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion (RRF) for combining multiple ranked lists.

    RRF score = sum(1 / (k + rank_i)) for each source i

    Paper: "Reciprocal Rank Fusion outperforms Condorcet and
           individual Rank Learning Methods" (Cormack et al., 2009)
    """

    def __init__(self, k: int = 60):
        """
        Initialize RRF.

        Args:
            k: Constant to prevent high scores for top-ranked items.
               Default 60 works well in practice.
        """
        self.k = k

    def fuse(
        self,
        ranked_lists: dict[str, list[tuple[str, float, any]]],
        top_k: int = 10,
    ) -> list[FusedResult]:
        """
        Fuse multiple ranked lists.

        Args:
            ranked_lists: Dict of source_name -> [(id, score, item), ...]
            top_k: Number of results to return

        Returns:
            List of FusedResult sorted by fused score
        """
        # Calculate RRF scores
        rrf_scores: dict[str, float] = defaultdict(float)
        items: dict[str, any] = {}
        source_scores: dict[str, dict[str, float]] = defaultdict(dict)

        for source_name, results in ranked_lists.items():
            for rank, (item_id, score, item) in enumerate(results, start=1):
                # RRF formula
                rrf_scores[item_id] += 1.0 / (self.k + rank)
                items[item_id] = item
                source_scores[item_id][source_name] = score

        # Sort by RRF score
        sorted_items = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        results = []
        for item_id, score in sorted_items[:top_k]:
            results.append(FusedResult(
                item=items[item_id],
                score=score,
                sources=dict(source_scores[item_id])
            ))

        return results


class LinearCombination:
    """
    Linear combination of scores with weights.

    score = sum(weight_i * normalized_score_i)
    """

    def __init__(self, weights: dict[str, float] = None):
        """
        Initialize with source weights.

        Args:
            weights: Dict of source_name -> weight (should sum to 1)
        """
        self.weights = weights or {}

    def fuse(
        self,
        ranked_lists: dict[str, list[tuple[str, float, any]]],
        top_k: int = 10,
    ) -> list[FusedResult]:
        """Fuse using linear combination."""
        # Normalize scores within each source
        normalized: dict[str, list[tuple[str, float, any]]] = {}

        for source_name, results in ranked_lists.items():
            if not results:
                continue

            scores = [score for _, score, _ in results]
            min_score = min(scores)
            max_score = max(scores)
            range_score = max_score - min_score or 1

            normalized[source_name] = [
                (item_id, (score - min_score) / range_score, item)
                for item_id, score, item in results
            ]

        # Combine scores
        combined: dict[str, float] = defaultdict(float)
        items: dict[str, any] = {}
        source_scores: dict[str, dict[str, float]] = defaultdict(dict)

        for source_name, results in normalized.items():
            weight = self.weights.get(source_name, 1.0 / len(normalized))

            for item_id, norm_score, item in results:
                combined[item_id] += weight * norm_score
                items[item_id] = item
                source_scores[item_id][source_name] = norm_score

        # Sort and return
        sorted_items = sorted(
            combined.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            FusedResult(
                item=items[item_id],
                score=score,
                sources=dict(source_scores[item_id])
            )
            for item_id, score in sorted_items[:top_k]
        ]
```

---

### Step 10: Hybrid Retriever

```python
# src/rag/hybrid/hybrid_retriever.py
"""Hybrid retriever combining vector and keyword search."""
from typing import Protocol
from dataclasses import dataclass

from rag.hybrid.bm25 import BM25Index, BM25Result
from rag.hybrid.fusion import ReciprocalRankFusion, FusedResult
from rag.embeddings.embedder import Embedder
from rag.vectorstore.qdrant import QdrantStore
from rag.retrieval.reranker import Reranker
from rag.models import Chunk
from rag.logging_config import get_logger

logger = get_logger("hybrid_retriever")


@dataclass
class HybridResult:
    """Result from hybrid search."""
    chunk: Chunk
    score: float
    vector_score: float | None
    keyword_score: float | None
    rerank_score: float | None = None


class HybridRetriever:
    """
    Hybrid retriever combining semantic and keyword search.

    Uses Reciprocal Rank Fusion to combine results from:
    - Vector search (semantic similarity)
    - BM25 (keyword matching)
    - Optionally: Reranking for final ordering
    """

    def __init__(
        self,
        embedder: Embedder = None,
        vector_store: QdrantStore = None,
        bm25_index: BM25Index = None,
        reranker: Reranker = None,
        fusion_k: int = 60,
        vector_weight: float = 0.5,
        keyword_weight: float = 0.5,
    ):
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or QdrantStore()
        self.bm25 = bm25_index or BM25Index()
        self.reranker = reranker
        self.fusion = ReciprocalRankFusion(k=fusion_k)
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Index chunks for keyword search."""
        self.bm25.index(chunks)

    async def search(
        self,
        query: str,
        top_k: int = 10,
        use_reranker: bool = False,
        alpha: float = None,  # Override vector weight
    ) -> list[HybridResult]:
        """
        Perform hybrid search.

        Args:
            query: Search query
            top_k: Number of results
            use_reranker: Whether to apply reranking
            alpha: Vector search weight (1-alpha for keyword)
        """
        # Get vector results
        query_vector = self.embedder.embed_query(query)
        vector_results = self.vector_store.search(
            query_vector,
            top_k=top_k * 2  # Fetch more for fusion
        )

        # Get keyword results
        keyword_results = self.bm25.search(query, top_k=top_k * 2)

        # Prepare for fusion
        ranked_lists = {
            "vector": [
                (chunk.id, score, chunk)
                for chunk, score in vector_results
            ],
            "keyword": [
                (result.chunk.id, result.score, result.chunk)
                for result in keyword_results
            ]
        }

        # Fuse results
        fused = self.fusion.fuse(ranked_lists, top_k=top_k * 2 if use_reranker else top_k)

        # Build results
        results = []
        for item in fused:
            results.append(HybridResult(
                chunk=item.item,
                score=item.score,
                vector_score=item.sources.get("vector"),
                keyword_score=item.sources.get("keyword"),
            ))

        # Rerank if enabled
        if use_reranker and self.reranker and results:
            reranked = self.reranker.rerank(
                query,
                [(r.chunk, r.score) for r in results],
                top_k=top_k
            )

            # Update with rerank scores
            reranked_results = []
            for chunk, rerank_score in reranked:
                # Find original result
                original = next((r for r in results if r.chunk.id == chunk.id), None)
                if original:
                    original.rerank_score = rerank_score
                    original.score = rerank_score  # Use rerank as final score
                    reranked_results.append(original)

            results = reranked_results

        logger.info(
            "hybrid_search",
            query=query[:50],
            vector_results=len(vector_results),
            keyword_results=len(keyword_results),
            fused_results=len(results)
        )

        return results[:top_k]
```

---

### Step 11: Conversation Memory

```python
# src/rag/conversation/memory.py
"""Conversation memory management."""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import json
import hashlib

from rag.cache.redis_client import RedisClient
from rag.logging_config import get_logger

logger = get_logger("conversation")


@dataclass
class Message:
    """A message in a conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ConversationContext:
    """Context from conversation history."""
    messages: list[Message]
    summary: str | None = None
    entities: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)


class ConversationMemory:
    """
    Manages conversation history with short-term and long-term memory.

    Short-term: Recent messages stored in Redis
    Long-term: Summarized history for context
    """

    def __init__(
        self,
        max_messages: int = 20,
        summary_threshold: int = 10,
        ttl_hours: int = 24,
    ):
        self.max_messages = max_messages
        self.summary_threshold = summary_threshold
        self.ttl = ttl_hours * 3600  # Convert to seconds

    def _get_key(self, session_id: str) -> str:
        """Get Redis key for session."""
        return f"conversation:{session_id}"

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict = None,
    ) -> None:
        """Add a message to conversation history."""
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {},
        )

        client = await RedisClient.get_client()
        key = self._get_key(session_id)

        # Get current messages
        messages = await self._get_messages(session_id)
        messages.append(message)

        # Check if we need to summarize
        if len(messages) > self.max_messages:
            # Keep only recent messages
            messages = messages[-self.max_messages:]

        # Store back
        await client.setex(
            key,
            self.ttl,
            json.dumps([m.to_dict() for m in messages])
        )

    async def get_context(
        self,
        session_id: str,
        max_messages: int = None,
    ) -> ConversationContext:
        """Get conversation context for a session."""
        messages = await self._get_messages(session_id)

        if max_messages:
            messages = messages[-max_messages:]

        # Extract entities and topics from recent messages
        entities = []
        topics = []
        for msg in messages[-5:]:
            entities.extend(msg.metadata.get("entities", []))
            topics.extend(msg.metadata.get("topics", []))

        return ConversationContext(
            messages=messages,
            entities=list(set(entities)),
            topics=list(set(topics)),
        )

    async def _get_messages(self, session_id: str) -> list[Message]:
        """Get messages for a session."""
        client = await RedisClient.get_client()
        key = self._get_key(session_id)

        data = await client.get(key)
        if not data:
            return []

        try:
            return [Message.from_dict(m) for m in json.loads(data)]
        except (json.JSONDecodeError, KeyError):
            return []

    async def clear(self, session_id: str) -> None:
        """Clear conversation history."""
        client = await RedisClient.get_client()
        await client.delete(self._get_key(session_id))

    def format_for_prompt(
        self,
        context: ConversationContext,
        max_tokens: int = 2000,
    ) -> str:
        """Format conversation history for inclusion in prompt."""
        lines = []
        total_chars = 0
        char_limit = max_tokens * 4  # Rough estimate

        # Add messages in reverse (most recent first for truncation)
        for msg in reversed(context.messages):
            line = f"{msg.role.capitalize()}: {msg.content}"
            if total_chars + len(line) > char_limit:
                break
            lines.insert(0, line)
            total_chars += len(line)

        return "\n".join(lines)


class ContextBuilder:
    """Build context for RAG queries from conversation."""

    def __init__(self, memory: ConversationMemory = None):
        self.memory = memory or ConversationMemory()

    async def build_query_context(
        self,
        session_id: str,
        current_query: str,
    ) -> dict:
        """
        Build context for a RAG query.

        Returns dict with:
        - rewritten_query: Query enhanced with context
        - conversation_context: Formatted history
        - entities: Mentioned entities
        """
        context = await self.memory.get_context(session_id, max_messages=5)

        # Check if query refers to previous context
        needs_rewrite = self._needs_context_rewrite(current_query, context)

        if needs_rewrite:
            rewritten = await self._rewrite_query(current_query, context)
        else:
            rewritten = current_query

        return {
            "rewritten_query": rewritten,
            "conversation_context": self.memory.format_for_prompt(context),
            "entities": context.entities,
            "original_query": current_query,
        }

    def _needs_context_rewrite(
        self,
        query: str,
        context: ConversationContext,
    ) -> bool:
        """Check if query needs context from history."""
        # Look for pronouns or references
        context_indicators = [
            "it", "this", "that", "these", "those",
            "they", "them", "he", "she", "the same",
            "previous", "earlier", "before", "again",
            "more", "another", "also", "too",
        ]

        query_lower = query.lower()
        return any(
            indicator in query_lower.split()
            for indicator in context_indicators
        )

    async def _rewrite_query(
        self,
        query: str,
        context: ConversationContext,
    ) -> str:
        """Rewrite query with context using LLM."""
        from rag.llm.groq import GroqLLM

        llm = GroqLLM()

        # Format recent conversation
        history = "\n".join([
            f"{m.role}: {m.content}"
            for m in context.messages[-4:]
        ])

        prompt = f"""Given the conversation history and the current query, rewrite the query to be self-contained (include all necessary context).

Conversation:
{history}

Current query: {query}

Rewritten query (just the query, nothing else):"""

        response = llm.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )

        return response.choices[0].message.content.strip()
```

---

### Step 12: HyDE (Hypothetical Document Embeddings)

```python
# src/rag/strategies/hyde.py
"""Hypothetical Document Embeddings (HyDE) retrieval strategy."""
from rag.embeddings.embedder import Embedder
from rag.vectorstore.qdrant import QdrantStore
from rag.llm.groq import GroqLLM
from rag.models import Chunk
from rag.logging_config import get_logger

logger = get_logger("hyde")


class HyDERetriever:
    """
    HyDE: Hypothetical Document Embeddings

    Instead of embedding the query directly, HyDE:
    1. Uses LLM to generate a hypothetical answer document
    2. Embeds the hypothetical document
    3. Uses that embedding for retrieval

    This often improves retrieval by bridging the gap between
    how questions and answers are written.

    Paper: "Precise Zero-Shot Dense Retrieval without Relevance Labels"
    """

    HYDE_PROMPT = """Write a short passage that would answer the following question.
Write as if you're writing a paragraph from a document that contains the answer.
Be specific and factual. Do not mention the question.

Question: {query}

Passage:"""

    def __init__(
        self,
        embedder: Embedder = None,
        store: QdrantStore = None,
        llm: GroqLLM = None,
        num_hypothetical: int = 1,
    ):
        self.embedder = embedder or Embedder()
        self.store = store or QdrantStore()
        self.llm = llm or GroqLLM()
        self.num_hypothetical = num_hypothetical

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[Chunk, float]]:
        """Retrieve using HyDE strategy."""
        # Generate hypothetical documents
        hypothetical_docs = await self._generate_hypothetical(query)

        logger.debug("hyde_generated", count=len(hypothetical_docs))

        # Embed hypothetical documents
        all_results = []
        for doc in hypothetical_docs:
            embedding = self.embedder.embed_query(doc)
            results = self.store.search(embedding, top_k=top_k)
            all_results.extend(results)

        # Deduplicate and re-rank by score
        seen_ids = set()
        unique_results = []
        for chunk, score in sorted(all_results, key=lambda x: x[1], reverse=True):
            if chunk.id not in seen_ids:
                seen_ids.add(chunk.id)
                unique_results.append((chunk, score))

        return unique_results[:top_k]

    async def _generate_hypothetical(self, query: str) -> list[str]:
        """Generate hypothetical answer documents."""
        prompt = self.HYDE_PROMPT.format(query=query)

        documents = []
        for _ in range(self.num_hypothetical):
            response = self.llm.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,  # Some variation
                max_tokens=300,
            )
            documents.append(response.choices[0].message.content)

        return documents
```

---

### Step 13: Query Decomposition

```python
# src/rag/strategies/decomposition.py
"""Query decomposition for complex questions."""
from dataclasses import dataclass
import json

from rag.llm.groq import GroqLLM
from rag.logging_config import get_logger

logger = get_logger("decomposition")


@dataclass
class SubQuery:
    """A decomposed sub-query."""
    query: str
    reasoning: str
    depends_on: list[int]  # Indices of dependent sub-queries


class QueryDecomposer:
    """
    Decompose complex queries into simpler sub-queries.

    Complex questions often require multiple pieces of information.
    This strategy:
    1. Breaks down the query into sub-questions
    2. Retrieves for each sub-question
    3. Combines results for final answer
    """

    DECOMPOSITION_PROMPT = """Break down the following complex question into simpler sub-questions that can be answered independently.

Question: {query}

Return a JSON array where each item has:
- "query": the sub-question
- "reasoning": why this sub-question is needed
- "depends_on": array of indices of sub-questions this depends on (empty if independent)

Example:
[
    {{"query": "What is X?", "reasoning": "Need to understand X first", "depends_on": []}},
    {{"query": "How does X relate to Y?", "reasoning": "Need to understand relationship", "depends_on": [0]}}
]

Sub-questions (JSON only):"""

    def __init__(self, llm: GroqLLM = None):
        self.llm = llm or GroqLLM()

    async def decompose(self, query: str) -> list[SubQuery]:
        """Decompose a query into sub-queries."""
        # Check if decomposition is needed
        if not self._needs_decomposition(query):
            return [SubQuery(query=query, reasoning="Simple query", depends_on=[])]

        prompt = self.DECOMPOSITION_PROMPT.format(query=query)

        response = self.llm.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1000,
        )

        content = response.choices[0].message.content

        try:
            # Parse JSON from response
            start = content.find("[")
            end = content.rfind("]") + 1
            data = json.loads(content[start:end])

            sub_queries = [
                SubQuery(
                    query=item["query"],
                    reasoning=item.get("reasoning", ""),
                    depends_on=item.get("depends_on", [])
                )
                for item in data
            ]

            logger.info("query_decomposed", original=query[:50], sub_queries=len(sub_queries))
            return sub_queries

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("decomposition_failed", error=str(e))
            return [SubQuery(query=query, reasoning="Decomposition failed", depends_on=[])]

    def _needs_decomposition(self, query: str) -> bool:
        """Heuristic to check if query needs decomposition."""
        # Complex queries often have:
        indicators = [
            " and ",  # Multiple topics
            " or ",
            " vs ",
            " versus ",
            " compare ",
            " difference ",
            " relationship ",
            " how does ",
            " why does ",
            " what are the ",
            " list ",
            " explain ",
        ]

        query_lower = query.lower()

        # Check for indicators
        has_indicator = any(ind in query_lower for ind in indicators)

        # Check query length (complex queries tend to be longer)
        is_long = len(query.split()) > 15

        # Check for multiple question marks
        multiple_questions = query.count("?") > 1

        return has_indicator or is_long or multiple_questions


class DecompositionRetriever:
    """Retriever that uses query decomposition."""

    def __init__(
        self,
        decomposer: QueryDecomposer = None,
        base_retriever=None,  # Any retriever with retrieve method
    ):
        self.decomposer = decomposer or QueryDecomposer()
        self.base_retriever = base_retriever

    async def retrieve(self, query: str, top_k: int = 5) -> list:
        """Retrieve using decomposition."""
        sub_queries = await self.decomposer.decompose(query)

        # Execute sub-queries in dependency order
        all_results = []
        sub_results = {}  # Cache results by index

        for i, sub_query in enumerate(sub_queries):
            # Wait for dependencies (in real impl, could parallelize independent ones)
            # For now, just retrieve for each

            results = await self.base_retriever.retrieve(
                sub_query.query,
                top_k=top_k
            )
            sub_results[i] = results
            all_results.extend(results)

        # Deduplicate
        seen_ids = set()
        unique_results = []
        for result in all_results:
            chunk = result[0] if isinstance(result, tuple) else result
            if chunk.id not in seen_ids:
                seen_ids.add(chunk.id)
                unique_results.append(result)

        return unique_results[:top_k]
```

---

### Step 14: Plugin Architecture

```python
# src/rag/plugins/base.py
"""Plugin system base classes and protocols."""
from typing import Protocol, runtime_checkable, Any, Callable
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum


class PluginType(Enum):
    """Types of plugins."""
    RETRIEVER = "retriever"
    EMBEDDER = "embedder"
    RERANKER = "reranker"
    PROCESSOR = "processor"
    TOOL = "tool"
    HOOK = "hook"


@dataclass
class PluginMetadata:
    """Plugin metadata."""
    name: str
    version: str
    description: str
    plugin_type: PluginType
    author: str = ""
    dependencies: list[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@runtime_checkable
class Plugin(Protocol):
    """Protocol for plugins."""

    @property
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        ...

    def initialize(self, config: dict) -> None:
        """Initialize the plugin with configuration."""
        ...

    def shutdown(self) -> None:
        """Clean up plugin resources."""
        ...


class BasePlugin:
    """Base class for plugins."""

    def __init__(self):
        self._metadata: PluginMetadata | None = None
        self._initialized = False

    @property
    def metadata(self) -> PluginMetadata:
        if self._metadata is None:
            raise NotImplementedError("Plugin must define metadata")
        return self._metadata

    def initialize(self, config: dict) -> None:
        """Initialize with config. Override in subclass."""
        self._initialized = True

    def shutdown(self) -> None:
        """Clean up. Override in subclass."""
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized
```

```python
# src/rag/plugins/registry.py
"""Plugin registry for managing loaded plugins."""
from typing import Type
from rag.plugins.base import Plugin, PluginType, PluginMetadata
from rag.logging_config import get_logger

logger = get_logger("plugins")


class PluginRegistry:
    """Registry for managing plugins."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._plugins = {}
            cls._instance._hooks = {}
        return cls._instance

    def register(self, plugin_class: Type[Plugin]) -> None:
        """Register a plugin class."""
        # Create temporary instance to get metadata
        instance = plugin_class()
        metadata = instance.metadata

        key = f"{metadata.plugin_type.value}:{metadata.name}"

        if key in self._plugins:
            logger.warning("plugin_override", name=metadata.name)

        self._plugins[key] = plugin_class
        logger.info("plugin_registered", name=metadata.name, type=metadata.plugin_type.value)

    def get(self, plugin_type: PluginType, name: str) -> Plugin | None:
        """Get a plugin instance."""
        key = f"{plugin_type.value}:{name}"
        plugin_class = self._plugins.get(key)

        if plugin_class:
            return plugin_class()
        return None

    def get_all(self, plugin_type: PluginType = None) -> list[Plugin]:
        """Get all plugins of a type."""
        plugins = []
        for key, plugin_class in self._plugins.items():
            if plugin_type is None or key.startswith(f"{plugin_type.value}:"):
                plugins.append(plugin_class())
        return plugins

    def list_plugins(self) -> list[PluginMetadata]:
        """List all registered plugin metadata."""
        return [
            plugin_class().metadata
            for plugin_class in self._plugins.values()
        ]

    def register_hook(self, hook_name: str, callback: callable) -> None:
        """Register a hook callback."""
        if hook_name not in self._hooks:
            self._hooks[hook_name] = []
        self._hooks[hook_name].append(callback)

    async def trigger_hook(self, hook_name: str, *args, **kwargs) -> list:
        """Trigger a hook and collect results."""
        results = []
        for callback in self._hooks.get(hook_name, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    result = await callback(*args, **kwargs)
                else:
                    result = callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error("hook_error", hook=hook_name, error=str(e))
        return results


# Decorator for registering plugins
def register_plugin(cls: Type[Plugin]) -> Type[Plugin]:
    """Decorator to register a plugin."""
    PluginRegistry().register(cls)
    return cls


# Global registry instance
registry = PluginRegistry()
```

```python
# src/rag/plugins/loader.py
"""Dynamic plugin loading."""
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Type

from rag.plugins.base import Plugin
from rag.plugins.registry import PluginRegistry
from rag.logging_config import get_logger

logger = get_logger("plugin_loader")


class PluginLoader:
    """Load plugins from files and directories."""

    def __init__(self, plugin_dirs: list[Path] = None):
        self.plugin_dirs = plugin_dirs or [Path("plugins")]
        self.registry = PluginRegistry()

    def discover_plugins(self) -> list[Path]:
        """Discover plugin files."""
        plugin_files = []

        for plugin_dir in self.plugin_dirs:
            if not plugin_dir.exists():
                continue

            # Look for plugin.py files in subdirectories
            for subdir in plugin_dir.iterdir():
                if subdir.is_dir():
                    plugin_file = subdir / "plugin.py"
                    if plugin_file.exists():
                        plugin_files.append(plugin_file)

            # Also look for direct .py files
            for file in plugin_dir.glob("*.py"):
                if file.name != "__init__.py":
                    plugin_files.append(file)

        return plugin_files

    def load_plugin(self, plugin_path: Path) -> Plugin | None:
        """Load a single plugin from path."""
        try:
            # Create module name from path
            module_name = f"rag_plugin_{plugin_path.stem}"

            # Load module from file
            spec = importlib.util.spec_from_file_location(
                module_name, plugin_path
            )
            if spec is None or spec.loader is None:
                logger.error("plugin_load_failed", path=str(plugin_path))
                return None

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Look for Plugin classes
            for name in dir(module):
                obj = getattr(module, name)
                if (
                    isinstance(obj, type)
                    and issubclass(obj, Plugin)
                    and obj is not Plugin
                    and hasattr(obj, "metadata")
                ):
                    # Register and return
                    self.registry.register(obj)
                    logger.info("plugin_loaded", name=name, path=str(plugin_path))
                    return obj()

        except Exception as e:
            logger.error("plugin_load_error", path=str(plugin_path), error=str(e))

        return None

    def load_all(self) -> list[Plugin]:
        """Load all discovered plugins."""
        plugins = []
        for plugin_path in self.discover_plugins():
            plugin = self.load_plugin(plugin_path)
            if plugin:
                plugins.append(plugin)

        logger.info("plugins_loaded", count=len(plugins))
        return plugins
```

---

### Step 15: A/B Testing Framework

```python
# src/rag/experiments/framework.py
"""A/B testing framework for RAG experiments."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Any
import random
import hashlib
import json

from rag.logging_config import get_logger

logger = get_logger("experiments")


@dataclass
class Variant:
    """Experiment variant."""
    name: str
    weight: float  # 0-1, probability of assignment
    config: dict = field(default_factory=dict)


@dataclass
class Experiment:
    """An A/B experiment."""
    id: str
    name: str
    description: str
    variants: list[Variant]
    start_date: datetime
    end_date: datetime | None = None
    is_active: bool = True

    def __post_init__(self):
        # Validate weights sum to 1
        total_weight = sum(v.weight for v in self.variants)
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"Variant weights must sum to 1, got {total_weight}")


@dataclass
class Assignment:
    """User assignment to experiment variant."""
    experiment_id: str
    variant_name: str
    user_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


class ExperimentFramework:
    """
    A/B testing framework for RAG improvements.

    Features:
    - Consistent user assignment (same user always gets same variant)
    - Weighted random assignment
    - Metric tracking per variant
    - Statistical significance testing
    """

    def __init__(self):
        self.experiments: dict[str, Experiment] = {}
        self.assignments: dict[str, Assignment] = {}  # user_id:exp_id -> assignment
        self.metrics: dict[str, list[dict]] = {}  # exp_id:variant -> metrics

    def create_experiment(
        self,
        name: str,
        description: str,
        variants: list[Variant],
        start_date: datetime = None,
        end_date: datetime = None,
    ) -> Experiment:
        """Create a new experiment."""
        exp_id = hashlib.md5(name.encode()).hexdigest()[:12]

        experiment = Experiment(
            id=exp_id,
            name=name,
            description=description,
            variants=variants,
            start_date=start_date or datetime.utcnow(),
            end_date=end_date,
        )

        self.experiments[exp_id] = experiment
        logger.info("experiment_created", exp_id=exp_id, name=name)

        return experiment

    def get_variant(
        self,
        experiment_id: str,
        user_id: str,
    ) -> Variant | None:
        """Get variant assignment for a user."""
        experiment = self.experiments.get(experiment_id)
        if not experiment or not experiment.is_active:
            return None

        # Check for existing assignment
        key = f"{user_id}:{experiment_id}"
        if key in self.assignments:
            variant_name = self.assignments[key].variant_name
            return next(
                (v for v in experiment.variants if v.name == variant_name),
                None
            )

        # Assign based on consistent hash
        variant = self._assign_variant(experiment, user_id)

        # Store assignment
        self.assignments[key] = Assignment(
            experiment_id=experiment_id,
            variant_name=variant.name,
            user_id=user_id,
        )

        logger.debug(
            "variant_assigned",
            exp_id=experiment_id,
            user_id=user_id,
            variant=variant.name
        )

        return variant

    def _assign_variant(self, experiment: Experiment, user_id: str) -> Variant:
        """Assign variant using consistent hashing."""
        # Hash user_id + experiment_id for consistent assignment
        hash_input = f"{user_id}:{experiment.id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        random_value = (hash_value % 10000) / 10000  # 0-1

        # Select variant based on weight
        cumulative = 0
        for variant in experiment.variants:
            cumulative += variant.weight
            if random_value < cumulative:
                return variant

        return experiment.variants[-1]

    def record_metric(
        self,
        experiment_id: str,
        variant_name: str,
        metric_name: str,
        value: float,
        metadata: dict = None,
    ) -> None:
        """Record a metric for a variant."""
        key = f"{experiment_id}:{variant_name}"
        if key not in self.metrics:
            self.metrics[key] = []

        self.metrics[key].append({
            "metric": metric_name,
            "value": value,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        })

    def get_results(self, experiment_id: str) -> dict:
        """Get experiment results with statistics."""
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            return {}

        results = {
            "experiment": {
                "id": experiment.id,
                "name": experiment.name,
                "start_date": experiment.start_date.isoformat(),
            },
            "variants": {},
        }

        for variant in experiment.variants:
            key = f"{experiment_id}:{variant.name}"
            metrics = self.metrics.get(key, [])

            # Aggregate metrics
            metric_values: dict[str, list[float]] = {}
            for m in metrics:
                metric_name = m["metric"]
                if metric_name not in metric_values:
                    metric_values[metric_name] = []
                metric_values[metric_name].append(m["value"])

            # Calculate statistics
            stats = {}
            for metric_name, values in metric_values.items():
                if values:
                    stats[metric_name] = {
                        "count": len(values),
                        "mean": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                    }

            results["variants"][variant.name] = {
                "weight": variant.weight,
                "sample_size": len(metrics),
                "metrics": stats,
            }

        return results


# Decorator for A/B testing functions
def ab_test(experiment_id: str, metric_name: str = "latency"):
    """Decorator to A/B test a function."""

    def decorator(func: Callable):
        async def wrapper(*args, user_id: str = None, **kwargs):
            framework = ExperimentFramework()

            # Get variant
            variant = framework.get_variant(experiment_id, user_id or "default")

            if variant:
                # Apply variant config
                kwargs.update(variant.config)

            # Execute and measure
            import time
            start = time.time()
            result = await func(*args, **kwargs)
            duration = time.time() - start

            # Record metric
            if variant:
                framework.record_metric(
                    experiment_id,
                    variant.name,
                    metric_name,
                    duration,
                    {"user_id": user_id}
                )

            return result

        return wrapper

    return decorator
```

---

### Step 16: Performance Profiling

```python
# src/rag/profiling/decorators.py
"""Performance profiling decorators."""
import time
import functools
import cProfile
import pstats
import io
from typing import Callable, Any
from contextlib import contextmanager

from rag.logging_config import get_logger
from rag.metrics.prometheus import (
    QUERY_LATENCY, EMBEDDING_LATENCY, LLM_LATENCY
)

logger = get_logger("profiling")


def timed(metric_histogram=None, log_threshold_ms: float = 1000):
    """
    Decorator to time function execution.

    Args:
        metric_histogram: Prometheus histogram to record to
        log_threshold_ms: Log warning if execution exceeds this
    """

    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                duration = (time.perf_counter() - start) * 1000  # ms

                if metric_histogram:
                    metric_histogram.observe(duration / 1000)  # Convert to seconds

                if duration > log_threshold_ms:
                    logger.warning(
                        "slow_function",
                        function=func.__name__,
                        duration_ms=duration
                    )
                else:
                    logger.debug(
                        "function_timed",
                        function=func.__name__,
                        duration_ms=duration
                    )

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration = (time.perf_counter() - start) * 1000

                if metric_histogram:
                    metric_histogram.observe(duration / 1000)

                if duration > log_threshold_ms:
                    logger.warning(
                        "slow_function",
                        function=func.__name__,
                        duration_ms=duration
                    )

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def profile(output_file: str = None, top_n: int = 20):
    """
    Decorator to profile a function with cProfile.

    Args:
        output_file: Optional file to save profile stats
        top_n: Number of top functions to log
    """

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            profiler.enable()

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                profiler.disable()

                # Format stats
                stream = io.StringIO()
                stats = pstats.Stats(profiler, stream=stream)
                stats.sort_stats("cumulative")
                stats.print_stats(top_n)

                logger.info(
                    "profile_results",
                    function=func.__name__,
                    stats=stream.getvalue()
                )

                if output_file:
                    stats.dump_stats(output_file)

        return wrapper

    return decorator


@contextmanager
def profile_block(name: str):
    """Context manager to profile a code block."""
    start = time.perf_counter()
    yield
    duration = (time.perf_counter() - start) * 1000
    logger.info("block_profiled", name=name, duration_ms=duration)


class PerformanceTracker:
    """Track performance metrics over time."""

    def __init__(self):
        self.metrics: dict[str, list[float]] = {}

    def record(self, name: str, value: float) -> None:
        """Record a metric value."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

    def get_stats(self, name: str) -> dict:
        """Get statistics for a metric."""
        values = self.metrics.get(name, [])
        if not values:
            return {}

        sorted_values = sorted(values)
        n = len(values)

        return {
            "count": n,
            "mean": sum(values) / n,
            "min": min(values),
            "max": max(values),
            "p50": sorted_values[n // 2],
            "p95": sorted_values[int(n * 0.95)] if n >= 20 else None,
            "p99": sorted_values[int(n * 0.99)] if n >= 100 else None,
        }

    def report(self) -> dict:
        """Generate full report."""
        return {
            name: self.get_stats(name)
            for name in self.metrics
        }
```

---

## Running Phase 4

### Local Development

```bash
# Start all infrastructure
docker-compose up -d postgres redis qdrant neo4j elasticsearch

# Run migrations
alembic upgrade head

# Initialize graph constraints
python -c "from rag.graph.graph_builder import KnowledgeGraphBuilder; import asyncio; asyncio.run(KnowledgeGraphBuilder().initialize())"

# Start API
uvicorn rag.api.main:app --reload --port 8000
```

### Example Usage

```python
# Using the Agent
from rag.agents.orchestrator import AgentOrchestrator
from rag.agents.tools.vector_search import VectorSearchTool

tools = [VectorSearchTool()]
agent = AgentOrchestrator(tools)

async for step in agent.run("What are the main features of the system?"):
    print(f"{step.state}: {step.thought}")
    if step.action:
        print(f"  Action: {step.action}")

# Using Hybrid Search
from rag.hybrid.hybrid_retriever import HybridRetriever

retriever = HybridRetriever()
results = await retriever.search("authentication API", top_k=5)

# Using Graph RAG
from rag.graph.graph_builder import KnowledgeGraphBuilder
from rag.graph.neo4j_client import Neo4jClient

builder = KnowledgeGraphBuilder()
await builder.process_document(document, chunks)

# Query related entities
client = Neo4jClient()
related = await client.find_related("authentication", depth=2)
```

---

## API Endpoints (New)

```python
# src/rag/api/routes/agents.py
"""Agent endpoints."""
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from typing import Annotated
import json

from rag.auth.dependencies import CurrentTenant
from rag.agents.orchestrator import AgentOrchestrator
from rag.agents.tools.vector_search import VectorSearchTool

router = APIRouter(prefix="/agent")


@router.post("/query")
async def agent_query(
    query: str,
    auth: CurrentTenant,
    stream: bool = False,
):
    """Query using agentic RAG."""
    tools = [VectorSearchTool(collection_name=f"tenant_{auth.tenant_id}_documents")]
    agent = AgentOrchestrator(tools)

    if stream:
        async def generate():
            async for step in agent.run(query):
                yield json.dumps({
                    "state": step.state.value,
                    "thought": step.thought,
                    "action": step.action,
                    "observation": step.observation,
                }) + "\n"

        return StreamingResponse(generate(), media_type="application/x-ndjson")

    # Non-streaming: collect all steps
    steps = []
    final_answer = None
    async for step in agent.run(query):
        steps.append({
            "state": step.state.value,
            "thought": step.thought,
            "action": step.action,
        })
        if step.action == "answer" and step.action_input:
            final_answer = step.action_input.get("answer")

    return {
        "answer": final_answer,
        "steps": steps,
    }
```

---

## Milestone Checklist

- [ ] Tool protocol and base classes
- [ ] Agent orchestrator with ReAct pattern
- [ ] Vector search tool for agents
- [ ] Neo4j client and connection
- [ ] Entity extraction (spaCy + LLM)
- [ ] Knowledge graph builder
- [ ] BM25 keyword search
- [ ] Reciprocal Rank Fusion
- [ ] Hybrid retriever
- [ ] Conversation memory
- [ ] HyDE retrieval strategy
- [ ] Query decomposition
- [ ] Plugin architecture with registry
- [ ] Plugin loader for dynamic loading
- [ ] A/B testing framework
- [ ] Performance profiling decorators
- [ ] Agent API endpoints
- [ ] Graph API endpoints
- [ ] Documentation updated

---

## Learning Checkpoints

After completing Phase 4, you should be able to answer:

- [ ] What is ReAct (Reasoning + Acting) and how do agents use it?
- [ ] How does knowledge graph augment vector search?
- [ ] When should you use hybrid search vs pure vector search?
- [ ] What is Reciprocal Rank Fusion and why is it effective?
- [ ] How does HyDE improve retrieval for certain queries?
- [ ] What are Protocol classes and how do they enable structural typing?
- [ ] How do you design a plugin system with dynamic loading?
- [ ] What metrics should you track in A/B tests for RAG systems?

---

## Next Steps

After completing Phase 4, consider:

1. **Production Hardening**: Rate limiting, circuit breakers, advanced monitoring
2. **Scaling**: Distributed agents, sharded vector stores
3. **Fine-tuning**: Custom embedding models, domain-specific rerankers
4. **Evaluation**: RAGAS metrics, automated testing pipelines

---

**Ready to start?** Begin with Step 1: Add New Dependencies!
