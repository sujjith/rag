"""
Phase 9.4: Message Queue Integration

Demonstrates:
- Apache Kafka integration
- RabbitMQ/Redis patterns
- Event-driven DAG triggers
- Message publishing
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.decorators import dag, task
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
import json


default_args = {
    "owner": "airflow",
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
}


@dag(
    dag_id="phase9_04_message_queues",
    description="Message queue integration patterns",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["phase9", "enterprise", "kafka", "messaging", "events"],
    doc_md="""
    ## Message Queue Integration

    **Kafka Prerequisites:**
    - Connection ID: `kafka_default`
    - Install: `apache-airflow-providers-apache-kafka`

    **RabbitMQ Prerequisites:**
    - Connection ID: `rabbitmq_default`

    **Patterns Covered:**
    - Kafka producer/consumer
    - Event publishing
    - Message-triggered DAGs
    - Dead letter queue handling
    """,
)
def message_queue_integration():

    # ==================== Kafka Operations ====================

    @task
    def produce_to_kafka():
        """Publish messages to Kafka topic."""
        topic = "data-events"
        messages = [
            {"event_type": "user_signup", "user_id": 123, "timestamp": datetime.now().isoformat()},
            {"event_type": "purchase", "user_id": 456, "amount": 99.99, "timestamp": datetime.now().isoformat()},
            {"event_type": "page_view", "user_id": 789, "page": "/home", "timestamp": datetime.now().isoformat()},
        ]

        print(f"Would publish to Kafka topic: {topic}")
        for msg in messages:
            print(f"  -> {json.dumps(msg)}")

        # In production:
        # from airflow.providers.apache.kafka.hooks.produce import KafkaProducerHook
        # hook = KafkaProducerHook(kafka_conn_id="kafka_default")
        # for msg in messages:
        #     hook.send_message(topic=topic, value=json.dumps(msg))

        return {"topic": topic, "messages_sent": len(messages)}

    @task
    def consume_from_kafka():
        """Consume messages from Kafka topic."""
        topic = "data-events"
        consumer_group = "airflow-consumer"

        print(f"Would consume from Kafka topic: {topic}")
        print(f"Consumer group: {consumer_group}")

        # In production:
        # from airflow.providers.apache.kafka.hooks.consume import KafkaConsumerHook
        # hook = KafkaConsumerHook(
        #     kafka_conn_id="kafka_default",
        #     topics=[topic],
        #     group_id=consumer_group,
        # )
        # messages = hook.consume(num_messages=100, timeout=30)

        # Simulated messages
        messages = [
            {"event_type": "user_signup", "user_id": 123},
            {"event_type": "purchase", "user_id": 456},
        ]

        return {"topic": topic, "messages_consumed": len(messages), "messages": messages}

    @task
    def process_kafka_batch(kafka_data: dict):
        """Process batch of Kafka messages."""
        messages = kafka_data.get("messages", [])

        # Group by event type
        events_by_type = {}
        for msg in messages:
            event_type = msg.get("event_type", "unknown")
            if event_type not in events_by_type:
                events_by_type[event_type] = []
            events_by_type[event_type].append(msg)

        print("Processed events by type:")
        for event_type, events in events_by_type.items():
            print(f"  {event_type}: {len(events)} events")

        return {
            "processed": len(messages),
            "by_type": {k: len(v) for k, v in events_by_type.items()},
        }

    # ==================== Event Publishing ====================

    @task
    def publish_pipeline_event(result: dict):
        """Publish pipeline completion event."""
        event = {
            "event_type": "pipeline_completed",
            "pipeline_id": "message_queue_integration",
            "status": "success",
            "records_processed": result.get("processed", 0),
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "dag_id": "phase9_04_message_queues",
                "run_id": "{{ run_id }}",
            }
        }

        topic = "pipeline-events"
        print(f"Would publish to {topic}:")
        print(json.dumps(event, indent=2))

        # In production:
        # hook = KafkaProducerHook(kafka_conn_id="kafka_default")
        # hook.send_message(topic=topic, value=json.dumps(event))

        return {"event_published": True, "topic": topic}

    # ==================== Dead Letter Queue ====================

    @task
    def handle_failed_messages():
        """Handle messages in dead letter queue."""
        dlq_topic = "data-events-dlq"

        print(f"Checking dead letter queue: {dlq_topic}")

        # Simulated DLQ messages
        failed_messages = [
            {
                "original_message": {"event_type": "malformed"},
                "error": "Schema validation failed",
                "retry_count": 3,
                "failed_at": "2024-01-01T10:00:00",
            }
        ]

        print(f"Found {len(failed_messages)} failed messages")

        # Processing strategies
        for msg in failed_messages:
            if msg["retry_count"] < 5:
                print(f"  -> Retrying message: {msg['original_message']}")
                # Republish to main topic
            else:
                print(f"  -> Moving to permanent failure: {msg['original_message']}")
                # Store in database for manual review

        return {"dlq_messages": len(failed_messages), "reprocessed": 0, "permanent_failures": 1}

    # ==================== Event-Driven Trigger ====================

    @task
    def check_for_trigger_events():
        """Check for events that should trigger other DAGs."""
        events = [
            {"type": "new_data_arrived", "source": "s3://bucket/data/"},
            {"type": "model_training_complete", "model_id": "model_v2"},
        ]

        triggers = []
        for event in events:
            if event["type"] == "new_data_arrived":
                triggers.append({
                    "dag_id": "data_processing_pipeline",
                    "conf": {"source": event["source"]},
                })
            elif event["type"] == "model_training_complete":
                triggers.append({
                    "dag_id": "model_deployment_pipeline",
                    "conf": {"model_id": event["model_id"]},
                })

        print(f"Would trigger {len(triggers)} DAGs:")
        for t in triggers:
            print(f"  -> {t['dag_id']} with conf: {t['conf']}")

        return {"triggers": triggers}

    # DAG flow
    produce = produce_to_kafka()
    consume = consume_from_kafka()
    processed = process_kafka_batch(consume)
    publish_pipeline_event(processed)

    handle_failed_messages()
    check_for_trigger_events()


message_queue_integration()
