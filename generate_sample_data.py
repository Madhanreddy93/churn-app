"""Creates a synthetic Telco-style churn dataset: data/sample_churn.csv"""
import os
import numpy as np, pandas as pd
os.makedirs("data", exist_ok=True)

rng = np.random.default_rng(42)
n = 3000
tenure = rng.integers(1, 73, n)
contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[.55, .25, .20])
internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[.35, .45, .20])
tech = rng.choice(["Yes", "No"], n, p=[.35, .65])
pay = rng.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n)
tickets = rng.poisson(1.5, n)
monthly = np.round(rng.normal(65, 25, n).clip(18, 120) + (internet == "Fiber optic") * 15, 2)
logins = rng.integers(0, 40, n)

z = (-0.6 - 0.035 * tenure + 1.1 * (contract == "Month-to-month") - 0.9 * (contract == "Two year")
     + 0.45 * (internet == "Fiber optic") - 0.5 * (tech == "Yes") + 0.3 * tickets
     + 0.01 * (monthly - 65) - 0.04 * logins + 0.4 * (pay == "Electronic check"))
churn = rng.random(n) < 1 / (1 + np.exp(-z))

df = pd.DataFrame({
    "customerID": [f"C{i:05d}" for i in range(n)], "tenure": tenure, "Contract": contract,
    "InternetService": internet, "TechSupport": tech, "PaymentMethod": pay,
    "SupportTickets": tickets, "MonthlyCharges": monthly, "MonthlyLogins": logins,
    "Churn": np.where(churn, "Yes", "No")})
df["TotalCharges"] = (df.tenure * df.MonthlyCharges).round(2)
df.to_csv("data/sample_churn.csv", index=False)
print(df.shape, "churn rate:", round(churn.mean(), 3))
