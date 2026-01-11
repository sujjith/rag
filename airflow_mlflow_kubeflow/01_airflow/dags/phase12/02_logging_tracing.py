"""
Phase 12.2: Structured Logging & Distributed Tracing

Demonstrates:
- Structured JSON logging
- Correlation IDs
- Distributed tracing (OpenTelemetry)
- Log aggregation patterns
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.decorators import dag, task
import json
import uuid
import logging


default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


class StructuredLogger:
    """
    Structured JSON logger for better log aggregation.
    Compatible with ELK Stack, Splunk, CloudWatch, etc.
    """

    def __init__(self, service: str, correlation_id: str = None):
        self.service = service
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.span_stack = []

    def _format(self, level: str, message: str, **kwargs) -> dict:
        """Format log entry as structured JSON."""
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level,
            "service": self.service,
            "correlation_id": self.correlation_id,
            "message": message,
            **kwargs
        }

        if self.span_stack:
            entry["span_id"] = self.span_stack[-1]

        return entry

    def info(self, message: str, **kwargs):
        """Log info level message."""
        entry = self._format("INFO", message, **kwargs)
        print(json.dumps(entry))

    def warning(self, message: str, **kwargs):
        """Log warning level message."""
        entry = self._format("WARNING", message, **kwargs)
        print(json.dumps(entry))

    def error(self, message: str, **kwargs):
        """Log error level message."""
        entry = self._format("ERROR", message, **kwargs)
        print(json.dumps(entry))

    def debug(self, message: str, **kwargs):
        """Log debug level message."""
        entry = self._format("DEBUG", message, **kwargs)
        print(json.dumps(entry))

    def start_span(self, name: str) -> str:
        """Start a new tracing span."""
        span_id = str(uuid.uuid4())[:8]
        self.span_stack.append(span_id)
        self.info(f"Span started: {name}", span_name=name, span_id=span_id)
        return span_id

    def end_span(self, name: str, status: str = "ok"):
        """End the current span."""
        if self.span_stack:
            span_id = self.span_stack.pop()
            self.info(f"Span ended: {name}", span_name=name, span_id=span_id, status=status)


class TraceContext:
    """
    Distributed tracing context.
    Passes trace information across tasks.
    """

    def __init__(self, trace_id: str = None, parent_span_id: str = None):
        self.trace_id = trace_id or str(uuid.uuid4())
        self.parent_span_id = parent_span_id
        self.spans = []

    def create_span(self, name: str, attributes: dict = None) -> dict:
        """Create a new span in the trace."""
        span = {
            "trace_id": self.trace_id,
            "span_id": str(uuid.uuid4())[:16],
            "parent_span_id": self.parent_span_id,
            "name": name,
            "start_time": datetime.utcnow().isoformat() + "Z",
            "attributes": attributes or {},
        }
        self.spans.append(span)
        return span

    def end_span(self, span: dict, status: str = "ok"):
        """End a span."""
        span["end_time"] = datetime.utcnow().isoformat() + "Z"
        span["status"] = status

    def to_dict(self) -> dict:
        """Export trace context."""
        return {
            "trace_id": self.trace_id,
            "spans": self.spans,
        }


@dag(
    dag_id="phase12_02_logging_tracing",
    description="Structured logging and distributed tracing",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["phase12", "enterprise", "logging", "tracing", "observability"],
    doc_md="""
    ## Structured Logging & Distributed Tracing

    **Structured Logging:**
    - JSON format for easy parsing
    - Correlation IDs for request tracing
    - Consistent field names
    - Contextual information

    **Distributed Tracing:**
    - OpenTelemetry integration
    - Span creation and propagation
    - Cross-service tracing
    - Performance analysis

    **Log Aggregation:**
    - ELK Stack (Elasticsearch, Logstash, Kibana)
    - Splunk
    - CloudWatch Logs
    - Datadog Logs
    """,
)
def logging_tracing():

    @task
    def initialize_trace():
        """Initialize trace context for the pipeline."""
        trace = TraceContext()
        logger = StructuredLogger("airflow-pipeline", trace.trace_id)

        logger.info("Pipeline started", dag_id="phase12_02_logging_tracing")

        return {
            "trace_id": trace.trace_id,
            "correlation_id": trace.trace_id,
        }

    @task
    def extract_with_logging(trace_context: dict):
        """Extract data with structured logging."""
        logger = StructuredLogger("extract-service", trace_context["correlation_id"])
        trace = TraceContext(trace_context["trace_id"])

        # Create span for extraction
        span = trace.create_span("data_extraction", {"source": "database"})

        logger.info("Starting data extraction", source="database", table="users")

        try:
            # Simulate extraction
            import time
            time.sleep(1)
            records = 1000

            logger.info(
                "Extraction completed",
                records_count=records,
                duration_ms=1000,
            )

            trace.end_span(span, status="ok")

        except Exception as e:
            logger.error(
                "Extraction failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            trace.end_span(span, status="error")
            raise

        return {
            "records": records,
            "trace": trace.to_dict(),
            "correlation_id": trace_context["correlation_id"],
        }

    @task
    def transform_with_logging(extract_result: dict):
        """Transform data with structured logging."""
        logger = StructuredLogger("transform-service", extract_result["correlation_id"])
        trace = TraceContext(extract_result["trace"]["trace_id"])

        span = trace.create_span("data_transformation", {"input_records": extract_result["records"]})

        logger.info(
            "Starting transformation",
            input_records=extract_result["records"],
        )

        # Simulate transformation with sub-operations
        operations = ["validate", "clean", "enrich", "format"]

        for op in operations:
            sub_span = trace.create_span(f"transform_{op}")
            logger.debug(f"Executing {op} operation", operation=op)

            import time
            time.sleep(0.5)

            trace.end_span(sub_span)

        output_records = int(extract_result["records"] * 0.95)

        logger.info(
            "Transformation completed",
            input_records=extract_result["records"],
            output_records=output_records,
            drop_rate=0.05,
        )

        trace.end_span(span)

        return {
            "records": output_records,
            "trace": trace.to_dict(),
            "correlation_id": extract_result["correlation_id"],
        }

    @task
    def load_with_logging(transform_result: dict):
        """Load data with structured logging."""
        logger = StructuredLogger("load-service", transform_result["correlation_id"])
        trace = TraceContext(transform_result["trace"]["trace_id"])

        span = trace.create_span("data_loading", {"destination": "warehouse"})

        logger.info(
            "Starting data load",
            records_count=transform_result["records"],
            destination="warehouse",
        )

        # Simulate batch loading
        batch_size = 100
        total = transform_result["records"]
        batches = (total + batch_size - 1) // batch_size

        for i in range(min(batches, 3)):  # Limit output for demo
            logger.debug(
                f"Loading batch {i+1}/{batches}",
                batch_number=i+1,
                batch_size=batch_size,
            )

        logger.info(
            "Load completed",
            records_loaded=total,
            batches=batches,
        )

        trace.end_span(span)

        return {
            "loaded": total,
            "trace": trace.to_dict(),
            "correlation_id": transform_result["correlation_id"],
        }

    @task
    def export_traces(load_result: dict):
        """Export trace data for visualization."""
        trace_data = load_result["trace"]

        print("\n" + "=" * 60)
        print("DISTRIBUTED TRACE")
        print("=" * 60)
        print(f"Trace ID: {trace_data['trace_id']}")
        print("\nSpans:")

        for span in trace_data["spans"]:
            indent = "  " if span.get("parent_span_id") else ""
            print(f"{indent}├── {span['name']}")
            print(f"{indent}│   Span ID: {span['span_id']}")
            print(f"{indent}│   Status: {span.get('status', 'unknown')}")
            if span.get("attributes"):
                print(f"{indent}│   Attributes: {span['attributes']}")

        # In production, export to:
        # - Jaeger
        # - Zipkin
        # - AWS X-Ray
        # - Datadog APM

        return {"exported": True, "span_count": len(trace_data["spans"])}

    @task
    def generate_log_summary(load_result: dict, trace_export: dict):
        """Generate summary of logging and tracing."""
        correlation_id = load_result["correlation_id"]

        summary = {
            "correlation_id": correlation_id,
            "trace_id": load_result["trace"]["trace_id"],
            "spans_created": trace_export["span_count"],
            "records_processed": load_result["loaded"],
            "status": "success",
        }

        print("\n" + "=" * 60)
        print("PIPELINE SUMMARY")
        print("=" * 60)
        print(json.dumps(summary, indent=2))

        # This summary would be:
        # 1. Sent to log aggregation system
        # 2. Used for dashboard metrics
        # 3. Stored for audit trail

        return summary

    # Pipeline flow
    trace = initialize_trace()
    extracted = extract_with_logging(trace)
    transformed = transform_with_logging(extracted)
    loaded = load_with_logging(transformed)
    trace_export = export_traces(loaded)
    generate_log_summary(loaded, trace_export)


logging_tracing()
