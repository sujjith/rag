"""
Phase 7.3: Circuit Breaker Pattern

Demonstrates:
- Circuit breaker for external services
- State management (closed, open, half-open)
- Automatic recovery
- Failure threshold management
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.exceptions import AirflowException
import json


default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(seconds=5),
}


class CircuitBreaker:
    """
    Circuit Breaker implementation using Airflow Variables.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failure threshold exceeded, requests blocked
    - HALF_OPEN: Testing if service recovered
    """

    def __init__(self, service_name: str, failure_threshold: int = 5,
                 recovery_timeout: int = 60):
        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.var_key = f"circuit_breaker_{service_name}"

    def _get_state(self) -> dict:
        """Get current circuit breaker state."""
        try:
            state = Variable.get(self.var_key, deserialize_json=True)
        except KeyError:
            state = {
                "status": "CLOSED",
                "failure_count": 0,
                "last_failure_time": None,
                "last_success_time": None,
            }
            self._set_state(state)
        return state

    def _set_state(self, state: dict):
        """Update circuit breaker state."""
        Variable.set(self.var_key, json.dumps(state))

    def can_execute(self) -> bool:
        """Check if request should be allowed."""
        state = self._get_state()

        if state["status"] == "CLOSED":
            return True

        if state["status"] == "OPEN":
            # Check if recovery timeout has passed
            if state["last_failure_time"]:
                last_failure = datetime.fromisoformat(state["last_failure_time"])
                if (datetime.now() - last_failure).seconds > self.recovery_timeout:
                    # Move to half-open state
                    state["status"] = "HALF_OPEN"
                    self._set_state(state)
                    print(f"Circuit {self.service_name}: OPEN -> HALF_OPEN")
                    return True
            return False

        if state["status"] == "HALF_OPEN":
            return True

        return False

    def record_success(self):
        """Record successful execution."""
        state = self._get_state()
        state["failure_count"] = 0
        state["last_success_time"] = datetime.now().isoformat()

        if state["status"] in ["OPEN", "HALF_OPEN"]:
            state["status"] = "CLOSED"
            print(f"Circuit {self.service_name}: {state['status']} -> CLOSED")

        self._set_state(state)

    def record_failure(self):
        """Record failed execution."""
        state = self._get_state()
        state["failure_count"] += 1
        state["last_failure_time"] = datetime.now().isoformat()

        if state["status"] == "HALF_OPEN":
            # Failed during test, back to open
            state["status"] = "OPEN"
            print(f"Circuit {self.service_name}: HALF_OPEN -> OPEN")
        elif state["failure_count"] >= self.failure_threshold:
            state["status"] = "OPEN"
            print(f"Circuit {self.service_name}: CLOSED -> OPEN (threshold reached)")

        self._set_state(state)

    def get_status(self) -> str:
        """Get current circuit status."""
        return self._get_state()["status"]


@dag(
    dag_id="phase7_03_circuit_breaker",
    description="Circuit breaker pattern for external services",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["phase7", "enterprise", "circuit-breaker", "resilience"],
    doc_md="""
    ## Circuit Breaker Pattern

    Prevents cascading failures when external services are down.

    **States:**
    - **CLOSED**: Normal operation, all requests pass
    - **OPEN**: Service is failing, requests are blocked
    - **HALF_OPEN**: Testing if service recovered

    **Configuration:**
    - `failure_threshold`: Failures before opening circuit
    - `recovery_timeout`: Seconds before testing recovery

    **Usage:**
    ```python
    cb = CircuitBreaker("my_api", failure_threshold=5, recovery_timeout=60)

    if cb.can_execute():
        try:
            result = call_api()
            cb.record_success()
        except Exception:
            cb.record_failure()
    else:
        # Use fallback or skip
        pass
    ```
    """,
)
def circuit_breaker_dag():

    @task
    def check_circuit_status():
        """Check current circuit breaker status."""
        cb = CircuitBreaker("external_api", failure_threshold=3, recovery_timeout=30)
        status = cb.get_status()
        print(f"Circuit breaker status: {status}")
        return {"status": status, "can_execute": cb.can_execute()}

    @task
    def call_external_service(circuit_info: dict):
        """Call external service with circuit breaker protection."""
        import random

        cb = CircuitBreaker("external_api", failure_threshold=3, recovery_timeout=30)

        if not cb.can_execute():
            print("Circuit is OPEN - skipping external call")
            return {"result": None, "source": "circuit_open", "skipped": True}

        try:
            # Simulate external API call
            print("Calling external API...")

            # 40% failure rate for demo
            if random.random() < 0.4:
                raise AirflowException("External API error")

            # Success
            cb.record_success()
            print("API call successful!")
            return {"result": "data_from_api", "source": "api", "skipped": False}

        except AirflowException as e:
            cb.record_failure()
            print(f"API call failed: {e}")

            # Return fallback instead of failing
            return {"result": "cached_data", "source": "fallback", "skipped": False}

    @task
    def call_with_fallback(api_result: dict):
        """Process result with fallback handling."""
        if api_result.get("skipped"):
            print("Using fallback due to open circuit")
            return {"processed": True, "data": "default_value"}

        if api_result.get("source") == "fallback":
            print("Using cached data due to API failure")

        return {"processed": True, "data": api_result.get("result")}

    @task
    def report_circuit_health():
        """Report circuit breaker health metrics."""
        cb = CircuitBreaker("external_api", failure_threshold=3, recovery_timeout=30)
        state = cb._get_state()

        print("=" * 50)
        print("CIRCUIT BREAKER HEALTH REPORT")
        print("=" * 50)
        print(f"Service: external_api")
        print(f"Status: {state['status']}")
        print(f"Failure Count: {state['failure_count']}")
        print(f"Last Failure: {state.get('last_failure_time', 'N/A')}")
        print(f"Last Success: {state.get('last_success_time', 'N/A')}")
        print("=" * 50)

        return state

    # DAG flow
    status = check_circuit_status()
    api_result = call_external_service(status)
    processed = call_with_fallback(api_result)
    report_circuit_health()


circuit_breaker_dag()
