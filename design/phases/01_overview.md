# RAG System - Overview

## Goals & Requirements

### Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1 | Ingest documents (PDF, TXT, MD, DOCX) | Must Have |
| FR-2 | Semantic search across documents | Must Have |
| FR-3 | Question answering with source citations | Must Have |
| FR-4 | REST API for all operations | Must Have |
| FR-5 | Multi-tenant data isolation | Must Have |
| FR-6 | Document-level access control | Should Have |
| FR-7 | Conversation history / multi-turn | Should Have |
| FR-8 | 50+ data source connectors | Nice to Have |

### Non-Functional Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-1 | Query latency (P95) | < 2 seconds |
| NFR-2 | Availability | 99.9% uptime |
| NFR-3 | Concurrent users | 1000+ |
| NFR-4 | Document scale | 10M+ chunks |
| NFR-5 | Security | SOC2, GDPR compliant |

---

## System Architecture

### Component Overview

```
                                    ┌─────────────────┐
                                    │   CDN / WAF     │
                                    │  (Cloudflare)   │
                                    └────────┬────────┘
                                             │
                                    ┌────────▼────────┐
                                    │  Load Balancer  │
                                    │   (Traefik)     │
                                    └────────┬────────┘
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    │                        │                        │
           ┌────────▼────────┐     ┌────────▼────────┐     ┌────────▼────────┐
           │   RAG API #1    │     │   RAG API #2    │     │   RAG API #3    │
           │   (FastAPI)     │     │   (FastAPI)     │     │   (FastAPI)     │
           └────────┬────────┘     └────────┬────────┘     └────────┬────────┘
                    │                        │                        │
                    └────────────────────────┼────────────────────────┘
                                             │
        ┌──────────────┬─────────────────────┼─────────────────────┬──────────────┐
        │              │                     │                     │              │
┌───────▼───────┐ ┌────▼────┐ ┌──────────────▼──────────────┐ ┌────▼────┐ ┌───────▼───────┐
│    Redis      │ │  Kafka  │ │      Qdrant Cluster         │ │Postgres │ │  Object Store │
│ (Cache/Queue) │ │ (Events)│ │  ┌─────┐ ┌─────┐ ┌─────┐   │ │(Metadata│ │     (S3)      │
└───────────────┘ └─────────┘ │  │Node1│ │Node2│ │Node3│   │ │ & Auth) │ │  (Documents)  │
                              │  └─────┘ └─────┘ └─────┘   │ └─────────┘ └───────────────┘
                              └────────────────────────────┘
                                             │
                              ┌──────────────┴──────────────┐
                              │                             │
                     ┌────────▼────────┐          ┌────────▼────────┐
                     │  Celery Worker  │          │  Celery Worker  │
                     │ (Doc Processing)│          │ (Doc Processing)│
                     └─────────────────┘          └─────────────────┘

        ┌─────────────────────────────────────────────────────────────┐
        │                    Observability Stack                       │
        │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
        │  │Prometheus│  │ Grafana  │  │  Jaeger  │  │   Loki   │    │
        │  │ (Metrics)│  │(Dashboard│  │ (Traces) │  │  (Logs)  │    │
        │  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
        └─────────────────────────────────────────────────────────────┘
```

---
