"""
Phase 9.2: API Integration

Demonstrates:
- REST API calls
- Authentication patterns
- Pagination handling
- Rate limiting
- Webhook handling
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.decorators import dag, task
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.providers.http.hooks.http import HttpHook
from airflow.models import Variable
import json
import time


default_args = {
    "owner": "airflow",
    "retries": 3,
    "retry_delay": timedelta(seconds=30),
    "retry_exponential_backoff": True,
}


class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, calls_per_minute: int = 60):
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute

    def wait(self):
        """Wait to respect rate limit."""
        time.sleep(self.min_interval)


@dag(
    dag_id="phase9_02_api_integration",
    description="REST API integration patterns",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["phase9", "enterprise", "api", "integration"],
    doc_md="""
    ## API Integration

    **Prerequisites:**
    1. Create HTTP connection in Airflow:
       - Connection ID: `api_default`
       - Connection Type: HTTP
       - Host: https://api.example.com
       - Extra: {"Authorization": "Bearer YOUR_TOKEN"}

    **Patterns Covered:**
    - REST API calls (GET, POST, PUT, DELETE)
    - OAuth2 authentication
    - API key authentication
    - Pagination handling
    - Rate limiting
    - Error handling and retries
    """,
)
def api_integration():

    @task
    def get_auth_token():
        """
        Get authentication token using OAuth2 client credentials.
        """
        # In production, credentials from Airflow Connection or Variable
        auth_config = {
            "client_id": "your_client_id",
            "client_secret": "your_client_secret",
            "token_url": "https://auth.example.com/oauth/token",
        }

        print(f"Would request token from: {auth_config['token_url']}")

        # In production:
        # import requests
        # response = requests.post(
        #     auth_config["token_url"],
        #     data={
        #         "grant_type": "client_credentials",
        #         "client_id": auth_config["client_id"],
        #         "client_secret": auth_config["client_secret"],
        #     }
        # )
        # token = response.json()["access_token"]

        # Simulated token
        return {
            "access_token": "simulated_token_12345",
            "expires_in": 3600,
            "token_type": "Bearer",
        }

    @task
    def fetch_paginated_data(auth: dict):
        """
        Fetch all data from paginated API endpoint.
        """
        base_url = "https://api.example.com/v1/records"
        headers = {"Authorization": f"Bearer {auth['access_token']}"}

        all_records = []
        page = 1
        per_page = 100
        rate_limiter = RateLimiter(calls_per_minute=30)

        # Simulate pagination
        total_pages = 3
        while page <= total_pages:
            url = f"{base_url}?page={page}&per_page={per_page}"
            print(f"Fetching: {url}")

            # In production:
            # hook = HttpHook(method="GET", http_conn_id="api_default")
            # response = hook.run(endpoint=f"/v1/records?page={page}&per_page={per_page}")
            # data = response.json()

            # Simulated response
            data = {
                "records": [{"id": i, "name": f"Record {i}"} for i in range((page-1)*10, page*10)],
                "pagination": {
                    "page": page,
                    "per_page": per_page,
                    "total_pages": total_pages,
                    "total_records": 30,
                }
            }

            all_records.extend(data["records"])
            print(f"Page {page}/{total_pages}: Got {len(data['records'])} records")

            page += 1
            rate_limiter.wait()

        return {"records": all_records, "total": len(all_records)}

    @task
    def post_data_to_api(auth: dict, data: dict):
        """
        POST data to API endpoint with retry logic.
        """
        endpoint = "/v1/records/batch"
        headers = {
            "Authorization": f"Bearer {auth['access_token']}",
            "Content-Type": "application/json",
        }

        payload = {
            "records": data["records"][:10],  # Batch of 10
            "metadata": {
                "source": "airflow",
                "timestamp": datetime.now().isoformat(),
            }
        }

        print(f"Would POST to: {endpoint}")
        print(f"Payload: {json.dumps(payload, indent=2)[:500]}...")

        # In production:
        # hook = HttpHook(method="POST", http_conn_id="api_default")
        # response = hook.run(
        #     endpoint=endpoint,
        #     data=json.dumps(payload),
        #     headers=headers
        # )

        # Simulated response
        return {
            "status": "success",
            "records_created": len(payload["records"]),
            "batch_id": "batch_12345",
        }

    @task
    def call_webhook(result: dict):
        """
        Call external webhook with pipeline results.
        """
        webhook_url = Variable.get("pipeline_webhook_url", default_var="https://hooks.example.com/pipeline")

        payload = {
            "event": "pipeline_completed",
            "pipeline": "api_integration",
            "status": result["status"],
            "records_processed": result["records_created"],
            "timestamp": datetime.now().isoformat(),
        }

        print(f"Would call webhook: {webhook_url}")
        print(f"Payload: {json.dumps(payload, indent=2)}")

        # In production:
        # import requests
        # response = requests.post(webhook_url, json=payload)

        return {"webhook_called": True, "status_code": 200}

    @task
    def handle_api_errors():
        """
        Demonstrate API error handling patterns.
        """
        error_handlers = {
            400: "Bad Request - Check payload format",
            401: "Unauthorized - Refresh token and retry",
            403: "Forbidden - Check permissions",
            404: "Not Found - Verify endpoint",
            429: "Rate Limited - Wait and retry with backoff",
            500: "Server Error - Retry with exponential backoff",
            502: "Bad Gateway - Retry after delay",
            503: "Service Unavailable - Retry with longer delay",
        }

        print("API Error Handling Strategy:")
        print("-" * 50)
        for code, action in error_handlers.items():
            print(f"  {code}: {action}")

        return {"error_handlers": list(error_handlers.keys())}

    # DAG flow
    auth = get_auth_token()
    data = fetch_paginated_data(auth)
    post_result = post_data_to_api(auth, data)
    call_webhook(post_result)
    handle_api_errors()


api_integration()
