"""Generate sample customer churn dataset"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)
n_customers = 1000

# Generate customer data
data = {
    'customer_id': range(1, n_customers + 1),
    'age': np.random.randint(18, 70, n_customers),
    'tenure_months': np.random.randint(1, 72, n_customers),
    'monthly_charges': np.random.uniform(20, 100, n_customers).round(2),
    'total_charges': np.random.uniform(100, 5000, n_customers).round(2),
    'num_support_tickets': np.random.randint(0, 10, n_customers),
    'num_products': np.random.randint(1, 5, n_customers),
    'payment_delay_days': np.random.randint(0, 30, n_customers),
    'contract_type': np.random.choice(['month-to-month', 'one-year', 'two-year'], n_customers),
    'internet_service': np.random.choice(['DSL', 'Fiber', 'None'], n_customers),
    'event_timestamp': [datetime.now() - timedelta(days=x) for x in range(n_customers)],
    'created_timestamp': [datetime.now()] * n_customers,
}

df = pd.DataFrame(data)

# Generate churn (higher for month-to-month, high support tickets, payment delays)
churn_prob = (
    0.1 + 
    (df['contract_type'] == 'month-to-month') * 0.2 +
    (df['num_support_tickets'] > 5) * 0.15 +
    (df['payment_delay_days'] > 15) * 0.1
)
df['churn'] = (np.random.random(n_customers) < churn_prob).astype(int)

# Save to CSV
df.to_csv('data/raw/customers.csv', index=False)
print(f"Generated {n_customers} customer records")
print(f"Churn rate: {df['churn'].mean():.1%}")
print(df.head())