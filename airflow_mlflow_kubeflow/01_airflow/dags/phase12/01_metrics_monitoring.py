"""
Phase 12.1: Metrics & Monitoring

Demonstrates:
- Custom metrics collection
- Prometheus integration
- Performance monitoring
- SLA tracking
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.decorators import dag, task
from airflow.models import TaskInstance
import time
import json


default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


class MetricsCollector:
    """
    Collect and export custom metrics.
    In production, integrate with Prometheus/StatsD.
    """

    def __init__(self):
        self.metrics = []

    def counter(self, name: str, value: int = 1, labels: dict = None):
        """Increment counter metric."""
        self.metrics.append({
            "type": "counter",
            "name": name,
            "value": value,
            "labels": labels or {},
            "timestamp": datetime.now().isoformat(),
        })

    def gauge(self, name: str, value: float, labels: dict = None):
        """Set gauge metric."""
        self.metrics.append({
            "type": "gauge",
            "name": name,
            "value": value,
            "labels": labels or {},
            "timestamp": datetime.now().isoformat(),
        })

    def histogram(self, name: str, value: float, labels: dict = None):
        """Record histogram observation."""
        self.metrics.append({
            "type": "histogram",
            "name": name,
            "value": value,
            "labels": labels or {},
            "timestamp": datetime.now().isoformat(),
        })

    def export(self) -> list:
        """Export all collected metrics."""
        return self.metrics


# Global metrics collector
metrics = MetricsCollector()


def record_task_metrics(context):
    """Callback to record task execution metrics."""
    ti = context["ti"]
    dag_id = context["dag"].dag_id
    task_id = ti.task_id

    # Calculate duration
    if ti.start_date and ti.end_date:
        duration = (ti.end_date - ti.start_date).total_seconds()
    else:
        duration = 0

    # Record metrics
    labels = {"dag_id": dag_id, "task_id": task_id}

    metrics.counter("airflow_task_completed_total", labels=labels)
    metrics.histogram("airflow_task_duration_seconds", duration, labels=labels)

    print(f"Recorded metrics for {dag_id}.{task_id}: duration={duration}s")


@dag(
    dag_id="phase12_01_metrics_monitoring",
    description="Custom metrics and monitoring patterns",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["phase12", "enterprise", "monitoring", "metrics"],
    doc_md="""
    ## Metrics & Monitoring

    **Metric Types:**
    - Counter: Cumulative count (e.g., tasks completed)
    - Gauge: Point-in-time value (e.g., queue length)
    - Histogram: Distribution of values (e.g., task duration)

    **Integration Options:**
    - Prometheus: prometheus-flask-exporter
    - StatsD: airflow.stats
    - DataDog: datadog agent
    - Custom endpoints

    **Key Metrics:**
    - Task duration
    - Task success/failure rate
    - Queue wait time
    - DAG run duration
    - Resource utilization
    """,
)
def metrics_monitoring():

    @task(on_success_callback=record_task_metrics)
    def data_extraction():
        """Extract data with timing metrics."""
        start_time = time.time()

        # Simulate work
        time.sleep(2)
        records_extracted = 1000

        duration = time.time() - start_time

        # Custom business metrics
        metrics.counter("records_extracted_total", records_extracted, {"source": "database"})
        metrics.gauge("extraction_records_per_second", records_extracted / duration)

        print(f"Extracted {records_extracted} records in {duration:.2f}s")
        return {"records": records_extracted, "duration": duration}

    @task(on_success_callback=record_task_metrics)
    def data_transformation(extract_result: dict):
        """Transform data with quality metrics."""
        start_time = time.time()

        # Simulate work
        time.sleep(1)

        total_records = extract_result["records"]
        valid_records = int(total_records * 0.95)  # 95% valid
        invalid_records = total_records - valid_records

        duration = time.time() - start_time

        # Quality metrics
        metrics.gauge("data_quality_valid_percentage", valid_records / total_records * 100)
        metrics.counter("records_invalid_total", invalid_records)
        metrics.histogram("transformation_duration_seconds", duration)

        print(f"Transformed {valid_records} valid records, {invalid_records} invalid")
        return {
            "valid_records": valid_records,
            "invalid_records": invalid_records,
            "duration": duration,
        }

    @task(on_success_callback=record_task_metrics)
    def data_loading(transform_result: dict):
        """Load data with throughput metrics."""
        start_time = time.time()

        # Simulate work
        time.sleep(1.5)
        records_loaded = transform_result["valid_records"]

        duration = time.time() - start_time
        throughput = records_loaded / duration

        # Loading metrics
        metrics.counter("records_loaded_total", records_loaded, {"destination": "warehouse"})
        metrics.gauge("load_throughput_records_per_second", throughput)

        print(f"Loaded {records_loaded} records at {throughput:.1f} records/s")
        return {"loaded": records_loaded, "throughput": throughput}

    @task
    def collect_pipeline_metrics(load_result: dict):
        """Collect and report pipeline-level metrics."""
        # Pipeline completion metric
        metrics.counter("pipeline_completed_total", labels={"status": "success"})

        # Export all metrics
        all_metrics = metrics.export()

        print("\n" + "=" * 60)
        print("COLLECTED METRICS")
        print("=" * 60)

        for m in all_metrics:
            labels_str = ", ".join(f"{k}={v}" for k, v in m.get("labels", {}).items())
            print(f"{m['type']:12} {m['name']:40} {m['value']:>10} [{labels_str}]")

        return {"metrics_count": len(all_metrics), "metrics": all_metrics}

    @task
    def export_to_prometheus(metrics_result: dict):
        """Export metrics in Prometheus format."""
        prometheus_output = []

        for m in metrics_result.get("metrics", []):
            name = m["name"]
            value = m["value"]
            labels = m.get("labels", {})
            labels_str = ",".join(f'{k}="{v}"' for k, v in labels.items())

            if labels_str:
                line = f"{name}{{{labels_str}}} {value}"
            else:
                line = f"{name} {value}"

            prometheus_output.append(line)

        print("\nPrometheus Format:")
        print("-" * 40)
        for line in prometheus_output:
            print(line)

        # In production, expose via /metrics endpoint
        return {"format": "prometheus", "lines": len(prometheus_output)}

    @task
    def check_sla_compliance():
        """Check SLA compliance and alert if violated."""
        sla_definitions = {
            "pipeline_duration_max_seconds": 300,
            "data_quality_min_percentage": 90,
            "throughput_min_records_per_second": 100,
        }

        # Get current values (simulated)
        current_values = {
            "pipeline_duration_max_seconds": 250,
            "data_quality_min_percentage": 95,
            "throughput_min_records_per_second": 150,
        }

        print("\nSLA Compliance Check:")
        print("-" * 40)

        violations = []
        for sla, threshold in sla_definitions.items():
            current = current_values.get(sla, 0)
            if "max" in sla:
                passed = current <= threshold
            else:
                passed = current >= threshold

            status = "✅" if passed else "❌"
            print(f"{status} {sla}: {current} (threshold: {threshold})")

            if not passed:
                violations.append(sla)
                metrics.counter("sla_violation_total", labels={"sla": sla})

        return {
            "compliant": len(violations) == 0,
            "violations": violations,
        }

    # Pipeline flow
    extracted = data_extraction()
    transformed = data_transformation(extracted)
    loaded = data_loading(transformed)
    metrics_result = collect_pipeline_metrics(loaded)
    export_to_prometheus(metrics_result)
    check_sla_compliance()


metrics_monitoring()
