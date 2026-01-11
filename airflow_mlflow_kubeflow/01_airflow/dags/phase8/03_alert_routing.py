"""
Phase 8.3: Alert Routing & Escalation

Demonstrates:
- Multi-channel alert routing
- Severity-based escalation
- On-call rotation integration
- Alert deduplication
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule
import json


default_args = {
    "owner": "airflow",
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
}


class AlertRouter:
    """
    Routes alerts to appropriate channels based on severity and context.

    Alert levels:
    - P1 (Critical): Page on-call, Slack #incidents, email leadership
    - P2 (High): Slack #alerts, email team
    - P3 (Medium): Slack #alerts
    - P4 (Low): Log only
    """

    # Alert routing configuration
    ROUTING_CONFIG = {
        "P1": {
            "channels": ["pagerduty", "slack_incidents", "email_leadership"],
            "description": "Critical - Immediate action required",
        },
        "P2": {
            "channels": ["slack_alerts", "email_team"],
            "description": "High - Action required within 1 hour",
        },
        "P3": {
            "channels": ["slack_alerts"],
            "description": "Medium - Action required today",
        },
        "P4": {
            "channels": ["log"],
            "description": "Low - Informational",
        },
    }

    # Team routing based on DAG tags
    TEAM_ROUTING = {
        "data-engineering": {
            "slack_channel": "#data-eng-alerts",
            "email": "data-eng@example.com",
            "pagerduty_service": "data-engineering",
        },
        "ml-platform": {
            "slack_channel": "#ml-alerts",
            "email": "ml-team@example.com",
            "pagerduty_service": "ml-platform",
        },
        "analytics": {
            "slack_channel": "#analytics-alerts",
            "email": "analytics@example.com",
            "pagerduty_service": "analytics",
        },
    }

    @classmethod
    def determine_severity(cls, context: dict) -> str:
        """Determine alert severity based on context."""
        dag_id = context.get("dag_id", "")
        task_id = context.get("task_id", "")
        try_number = context.get("try_number", 1)
        is_production = "prod" in dag_id.lower()

        # P1: Production failures after all retries
        if is_production and try_number > 2:
            return "P1"

        # P2: Production first failure or important tasks
        if is_production or "critical" in task_id.lower():
            return "P2"

        # P3: Non-production failures
        if try_number > 1:
            return "P3"

        # P4: First attempt failures in dev/test
        return "P4"

    @classmethod
    def get_team(cls, dag_tags: list) -> str:
        """Determine responsible team from DAG tags."""
        for tag in dag_tags:
            if tag in cls.TEAM_ROUTING:
                return tag
        return "data-engineering"  # Default team

    @classmethod
    def route_alert(cls, context: dict, severity: str = None) -> dict:
        """Route alert to appropriate channels."""
        if severity is None:
            severity = cls.determine_severity(context)

        routing = cls.ROUTING_CONFIG.get(severity, cls.ROUTING_CONFIG["P4"])
        team = cls.get_team(context.get("tags", []))
        team_config = cls.TEAM_ROUTING.get(team, cls.TEAM_ROUTING["data-engineering"])

        alert_info = {
            "severity": severity,
            "description": routing["description"],
            "channels": routing["channels"],
            "team": team,
            "team_config": team_config,
            "dag_id": context.get("dag_id"),
            "task_id": context.get("task_id"),
            "timestamp": datetime.now().isoformat(),
        }

        # Send to each channel
        for channel in routing["channels"]:
            cls._send_to_channel(channel, alert_info, team_config)

        return alert_info

    @classmethod
    def _send_to_channel(cls, channel: str, alert_info: dict, team_config: dict):
        """Send alert to specific channel."""
        print(f"Sending to {channel}:")

        if channel == "pagerduty":
            print(f"  -> PagerDuty service: {team_config['pagerduty_service']}")
            print(f"  -> Severity: {alert_info['severity']}")
            # In production: call PagerDuty API

        elif channel == "slack_incidents":
            print(f"  -> Slack #incidents channel")
            print(f"  -> @here notification for {alert_info['severity']}")
            # In production: send to incidents channel

        elif channel == "slack_alerts":
            print(f"  -> Slack {team_config['slack_channel']}")
            # In production: send to team's alert channel

        elif channel == "email_team":
            print(f"  -> Email: {team_config['email']}")
            # In production: send email

        elif channel == "email_leadership":
            print(f"  -> Email: leadership@example.com")
            # In production: send to leadership

        elif channel == "log":
            print(f"  -> Logged only (no notification)")


def alert_callback(context):
    """Unified alert callback with routing."""
    alert_context = {
        "dag_id": context["dag"].dag_id,
        "task_id": context["ti"].task_id,
        "try_number": context["ti"].try_number,
        "tags": list(context["dag"].tags),
        "exception": str(context.get("exception", "Unknown")),
    }

    result = AlertRouter.route_alert(alert_context)
    print(f"\nAlert routed: {result['severity']} to {result['channels']}")


@dag(
    dag_id="phase8_03_alert_routing",
    description="Multi-channel alert routing and escalation",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["phase8", "enterprise", "notifications", "data-engineering"],
    doc_md="""
    ## Alert Routing & Escalation

    **Severity Levels:**
    - P1: Critical - Pages on-call, Slack incidents, email leadership
    - P2: High - Slack alerts, email team
    - P3: Medium - Slack alerts only
    - P4: Low - Log only

    **Routing Logic:**
    - Production failures after retries -> P1
    - Production first failure -> P2
    - Development failures -> P3/P4

    **Team Routing:**
    - Based on DAG tags (data-engineering, ml-platform, analytics)
    - Each team has dedicated channels
    """,
)
def alert_routing():

    @task(on_failure_callback=alert_callback)
    def critical_production_task():
        """Simulates critical production task."""
        import random
        if random.random() < 0.3:
            raise Exception("Critical database connection failed")
        return {"status": "success"}

    @task(on_failure_callback=alert_callback)
    def normal_task():
        """Normal processing task."""
        import random
        if random.random() < 0.3:
            raise Exception("Processing error")
        return {"status": "success"}

    @task
    def demonstrate_routing():
        """Demonstrate alert routing for different scenarios."""
        scenarios = [
            {
                "dag_id": "prod_etl_pipeline",
                "task_id": "load_data",
                "try_number": 3,
                "tags": ["data-engineering", "production"],
            },
            {
                "dag_id": "prod_ml_training",
                "task_id": "train_model",
                "try_number": 1,
                "tags": ["ml-platform", "production"],
            },
            {
                "dag_id": "dev_analytics_report",
                "task_id": "generate_report",
                "try_number": 1,
                "tags": ["analytics", "development"],
            },
        ]

        print("\n" + "=" * 60)
        print("ALERT ROUTING DEMONSTRATION")
        print("=" * 60)

        for scenario in scenarios:
            print(f"\nScenario: {scenario['dag_id']}.{scenario['task_id']}")
            print("-" * 40)
            result = AlertRouter.route_alert(scenario)
            print(f"Result: {result['severity']} -> {result['channels']}")

        return "demonstration_complete"

    @task(trigger_rule=TriggerRule.ALL_DONE)
    def summarize():
        """Summarize alert routing results."""
        print("\nAlert routing demonstration complete")
        print("Check logs for routing decisions")
        return "done"

    # Flow
    t1 = critical_production_task()
    t2 = normal_task()
    demo = demonstrate_routing()
    summarize()


alert_routing()
