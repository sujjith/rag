"""
Phase 10.2: Schema Validation

Demonstrates:
- JSON Schema validation
- Pydantic models
- Schema evolution handling
- Schema registry patterns
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.decorators import dag, task
from airflow.exceptions import AirflowException
from typing import List, Optional
import json


default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


# Pydantic-style schema definitions (simplified for demo)
SCHEMAS = {
    "user_event_v1": {
        "type": "object",
        "required": ["event_id", "user_id", "event_type", "timestamp"],
        "properties": {
            "event_id": {"type": "string", "pattern": "^evt_[a-z0-9]+$"},
            "user_id": {"type": "integer", "minimum": 1},
            "event_type": {"type": "string", "enum": ["signup", "login", "purchase", "logout"]},
            "timestamp": {"type": "string", "format": "date-time"},
            "metadata": {"type": "object"},
        }
    },
    "user_event_v2": {
        "type": "object",
        "required": ["event_id", "user_id", "event_type", "timestamp", "source"],
        "properties": {
            "event_id": {"type": "string", "pattern": "^evt_[a-z0-9]+$"},
            "user_id": {"type": "integer", "minimum": 1},
            "event_type": {"type": "string", "enum": ["signup", "login", "purchase", "logout", "view"]},
            "timestamp": {"type": "string", "format": "date-time"},
            "source": {"type": "string", "enum": ["web", "mobile", "api"]},
            "metadata": {"type": "object"},
        }
    }
}


class SchemaValidator:
    """Simple JSON Schema validator."""

    def __init__(self, schema: dict):
        self.schema = schema

    def validate(self, data: dict) -> dict:
        """Validate data against schema."""
        errors = []

        # Check required fields
        for field in self.schema.get("required", []):
            if field not in data:
                errors.append(f"Missing required field: {field}")

        # Check field types and constraints
        properties = self.schema.get("properties", {})
        for field, value in data.items():
            if field in properties:
                field_schema = properties[field]
                field_errors = self._validate_field(field, value, field_schema)
                errors.extend(field_errors)

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "data": data,
        }

    def _validate_field(self, field: str, value, schema: dict) -> list:
        """Validate individual field."""
        errors = []

        # Type check
        expected_type = schema.get("type")
        if expected_type:
            type_map = {
                "string": str,
                "integer": int,
                "number": (int, float),
                "boolean": bool,
                "object": dict,
                "array": list,
            }
            expected = type_map.get(expected_type)
            if expected and not isinstance(value, expected):
                errors.append(f"Field {field}: expected {expected_type}, got {type(value).__name__}")

        # Enum check
        if "enum" in schema and value not in schema["enum"]:
            errors.append(f"Field {field}: value '{value}' not in allowed values {schema['enum']}")

        # Min/max for numbers
        if isinstance(value, (int, float)):
            if "minimum" in schema and value < schema["minimum"]:
                errors.append(f"Field {field}: value {value} below minimum {schema['minimum']}")
            if "maximum" in schema and value > schema["maximum"]:
                errors.append(f"Field {field}: value {value} above maximum {schema['maximum']}")

        return errors


@dag(
    dag_id="phase10_02_schema_validation",
    description="Schema validation patterns",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["phase10", "enterprise", "data-quality", "schema"],
    doc_md="""
    ## Schema Validation

    **Patterns Covered:**
    - JSON Schema validation
    - Schema versioning
    - Schema evolution
    - Forward/backward compatibility
    - Schema registry integration

    **Best Practices:**
    - Always version schemas
    - Support backward compatibility
    - Document breaking changes
    - Use schema registry for shared schemas
    """,
)
def schema_validation():

    @task
    def get_incoming_data():
        """Simulate incoming data with mixed schema versions."""
        data = [
            # Valid v1 events
            {
                "event_id": "evt_abc123",
                "user_id": 1,
                "event_type": "signup",
                "timestamp": "2024-01-01T10:00:00Z",
            },
            # Valid v2 events (with source)
            {
                "event_id": "evt_def456",
                "user_id": 2,
                "event_type": "login",
                "timestamp": "2024-01-01T11:00:00Z",
                "source": "web",
            },
            # Invalid event (missing required field)
            {
                "event_id": "evt_ghi789",
                "event_type": "purchase",
                "timestamp": "2024-01-01T12:00:00Z",
            },
            # Invalid event (wrong type)
            {
                "event_id": "evt_jkl012",
                "user_id": "not_an_integer",
                "event_type": "logout",
                "timestamp": "2024-01-01T13:00:00Z",
            },
            # Invalid event (unknown event_type for v1)
            {
                "event_id": "evt_mno345",
                "user_id": 5,
                "event_type": "view",
                "timestamp": "2024-01-01T14:00:00Z",
            },
        ]

        return {"events": data, "count": len(data)}

    @task
    def detect_schema_version(event: dict) -> str:
        """Detect schema version from event structure."""
        # Simple version detection based on field presence
        if "source" in event:
            return "user_event_v2"
        return "user_event_v1"

    @task
    def validate_with_schema(incoming: dict):
        """Validate each event against appropriate schema."""
        events = incoming["events"]
        results = {
            "valid_events": [],
            "invalid_events": [],
            "validation_details": [],
        }

        for event in events:
            # Detect and validate against appropriate version
            schema_version = "user_event_v2" if "source" in event else "user_event_v1"
            schema = SCHEMAS[schema_version]
            validator = SchemaValidator(schema)

            validation = validator.validate(event)
            validation["schema_version"] = schema_version

            if validation["valid"]:
                results["valid_events"].append(event)
            else:
                results["invalid_events"].append({
                    "event": event,
                    "errors": validation["errors"],
                    "schema_version": schema_version,
                })

            results["validation_details"].append(validation)

        print(f"Validation complete:")
        print(f"  Valid: {len(results['valid_events'])}")
        print(f"  Invalid: {len(results['invalid_events'])}")

        return results

    @task
    def handle_schema_evolution(validation_results: dict):
        """Handle schema evolution - upgrade v1 events to v2."""
        valid_events = validation_results["valid_events"]
        upgraded = []

        for event in valid_events:
            # Check if v1 event needs upgrade
            if "source" not in event:
                # Add default source for v1 events
                event["source"] = "unknown"
                event["_upgraded_from"] = "v1"
                print(f"Upgraded event {event['event_id']} from v1 to v2")

            upgraded.append(event)

        return {"events": upgraded, "upgrades": sum(1 for e in upgraded if "_upgraded_from" in e)}

    @task
    def handle_invalid_records(validation_results: dict):
        """Handle invalid records - dead letter queue pattern."""
        invalid = validation_results["invalid_events"]

        if not invalid:
            print("No invalid records to handle")
            return {"handled": 0}

        print(f"Handling {len(invalid)} invalid records:")
        for record in invalid:
            print(f"  Event: {record['event'].get('event_id', 'unknown')}")
            print(f"  Errors: {record['errors']}")
            print(f"  Action: Sending to dead letter queue")

        # In production:
        # - Write to DLQ table/topic
        # - Send alert if threshold exceeded
        # - Log for monitoring

        return {
            "handled": len(invalid),
            "action": "sent_to_dlq",
            "events": [r["event"].get("event_id") for r in invalid],
        }

    @task
    def validate_schema_compatibility():
        """Check schema compatibility for evolution."""
        v1 = SCHEMAS["user_event_v1"]
        v2 = SCHEMAS["user_event_v2"]

        compatibility_report = {
            "backward_compatible": True,
            "forward_compatible": False,
            "breaking_changes": [],
            "new_fields": [],
            "removed_fields": [],
        }

        # Check for removed required fields (breaking change)
        v1_required = set(v1.get("required", []))
        v2_required = set(v2.get("required", []))

        removed = v1_required - v2_required
        added = v2_required - v1_required

        if removed:
            compatibility_report["backward_compatible"] = False
            compatibility_report["breaking_changes"].append(f"Removed required fields: {removed}")

        if added:
            compatibility_report["new_fields"].extend(list(added))
            # New required fields break forward compatibility
            compatibility_report["breaking_changes"].append(f"New required fields: {added}")

        print("Schema Compatibility Report:")
        print(json.dumps(compatibility_report, indent=2))

        return compatibility_report

    # DAG flow
    data = get_incoming_data()
    validated = validate_with_schema(data)
    evolved = handle_schema_evolution(validated)
    handle_invalid_records(validated)
    validate_schema_compatibility()


schema_validation()
