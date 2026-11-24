"""Rule-based retention suggestions. Rules only fire if the column exists."""
import pandas as pd

def reasons_and_actions(row: pd.Series, df_cols) -> tuple[list[str], list[str]]:
    why, act = [], []
    has = lambda c: c in df_cols
    if has("Contract") and row["Contract"] == "Month-to-month":
        why.append("Month-to-month contract")
        act.append("Offer a discounted 1-year plan")
    if has("tenure") and row["tenure"] <= 6:
        why.append("New customer (<= 6 months)")
        act.append("Run an onboarding call and a 30-day check-in")
    if has("SupportTickets") and row["SupportTickets"] >= 3:
        why.append("Frequent support tickets")
        act.append("Escalate to a senior agent and follow up within 48 hours")
    if has("MonthlyLogins") and row["MonthlyLogins"] <= 5:
        why.append("Low product usage")
        act.append("Send a usage-tips campaign and highlight unused features")
    if has("TechSupport") and row["TechSupport"] == "No":
        why.append("No tech support add-on")
        act.append("Offer a free 3-month tech support trial")
    if has("MonthlyCharges") and row["MonthlyCharges"] >= 90:
        why.append("High monthly bill")
        act.append("Review the plan and offer a loyalty discount or right-size the plan")
    if has("PaymentMethod") and row["PaymentMethod"] == "Electronic check":
        why.append("Manual payment method")
        act.append("Encourage auto-pay with a small credit")
    if not act:
        why.append("Model-driven risk (no single rule)")
        act.append("Assign a customer success manager for a personal check-in")
    return why, act
