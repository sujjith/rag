"""
Phase 8.2: Slack Notifications

Demonstrates:
- Slack webhook integration
- Channel-based alerts
- Rich message formatting
- Alert severity levels
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.decorators import dag, task
from airflow.operators.empty import EmptyOperator
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
from airflow.providers.slack.hooks.slack_webhook import SlackWebhookHook
from airflow.utils.trigger_rule import TriggerRule
import json


default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


def send_slack_alert(context, severity="error"):
    """
    Send Slack alert with severity-based formatting.

    Severity levels:
    - info: Blue
    - warning: Yellow
    - error: Red
    - success: Green
    """
    colors = {
        "info": "#0066cc",
        "warning": "#ffcc00",
        "error": "#cc0000",
        "success": "#00cc00",
    }

    ti = context.get("ti")
    dag_id = context["dag"].dag_id
    task_id = ti.task_id if ti else "N/A"
    execution_date = context.get("execution_date", "N/A")
    exception = context.get("exception", "No exception details")
    log_url = ti.log_url if ti else ""

    # Slack message payload
    slack_msg = {
        "attachments": [
            {
                "color": colors.get(severity, colors["info"]),
                "blocks": [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": f"{'🚨' if severity == 'error' else '⚠️' if severity == 'warning' else 'ℹ️'} Airflow Alert",
                        }
                    },
                    {
                        "type": "section",
                        "fields": [
                            {"type": "mrkdwn", "text": f"*DAG:*\n{dag_id}"},
                            {"type": "mrkdwn", "text": f"*Task:*\n{task_id}"},
                            {"type": "mrkdwn", "text": f"*Severity:*\n{severity.upper()}"},
                            {"type": "mrkdwn", "text": f"*Date:*\n{execution_date}"},
                        ]
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*Details:*\n```{str(exception)[:500]}```"
                        }
                    },
                    {
                        "type": "actions",
                        "elements": [
                            {
                                "type": "button",
                                "text": {"type": "plain_text", "text": "View Logs"},
                                "url": log_url,
                            }
                        ]
                    }
                ]
            }
        ]
    }

    print(f"Slack Alert ({severity}): {dag_id}.{task_id}")
    print(json.dumps(slack_msg, indent=2))

    # In production with Slack webhook configured:
    # hook = SlackWebhookHook(slack_webhook_conn_id="slack_webhook")
    # hook.send_dict(slack_msg)


def slack_failure_callback(context):
    """Callback for task failures."""
    send_slack_alert(context, severity="error")


def slack_success_callback(context):
    """Callback for task success."""
    send_slack_alert(context, severity="success")


@dag(
    dag_id="phase8_02_slack_notifications",
    description="Slack notification patterns",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["phase8", "enterprise", "notifications", "slack"],
    doc_md="""
    ## Slack Notifications

    **Prerequisites:**
    1. Create Slack App with Incoming Webhooks
    2. Configure Airflow Connection:
       - Connection ID: `slack_webhook`
       - Connection Type: Slack Webhook
       - Host: https://hooks.slack.com/services
       - Password: T00000000/B00000000/XXXXXXXXXXXXXXX

    **Features:**
    - Severity-based color coding
    - Rich Block Kit formatting
    - Direct links to logs
    - Channel routing by alert type

    **Severity Levels:**
    - 🔵 Info: General notifications
    - 🟡 Warning: Non-critical issues
    - 🔴 Error: Task failures
    - 🟢 Success: Completion confirmations
    """,
)
def slack_notifications():

    @task(on_failure_callback=slack_failure_callback)
    def extract_data():
        """Extract data with Slack alert on failure."""
        import random

        if random.random() < 0.3:
            raise Exception("Failed to connect to data source")

        return {"records": 1000, "source": "database"}

    @task(on_failure_callback=slack_failure_callback)
    def transform_data(data: dict):
        """Transform with failure alerting."""
        import random

        if random.random() < 0.2:
            raise Exception("Data transformation error: invalid schema")

        return {"records": data["records"], "transformed": True}

    @task(
        on_failure_callback=slack_failure_callback,
        on_success_callback=slack_success_callback,
    )
    def load_data(data: dict):
        """Load with success/failure alerts."""
        print(f"Loading {data['records']} records")
        return {"loaded": True, "records": data["records"]}

    @task
    def send_summary_notification(result: dict):
        """Send pipeline summary to Slack."""
        summary = {
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "✅ Pipeline Completed Successfully"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Records Processed:*\n{result['records']}"},
                        {"type": "mrkdwn", "text": "*Status:*\n✅ Success"},
                    ]
                },
                {
                    "type": "context",
                    "elements": [
                        {"type": "mrkdwn", "text": f"Completed at {datetime.now().isoformat()}"}
                    ]
                }
            ]
        }

        print("Summary Notification:")
        print(json.dumps(summary, indent=2))

        # In production:
        # hook = SlackWebhookHook(slack_webhook_conn_id="slack_webhook")
        # hook.send_dict(summary)

        return "notification_sent"

    # Pipeline flow
    data = extract_data()
    transformed = transform_data(data)
    loaded = load_data(transformed)
    send_summary_notification(loaded)


slack_notifications()
