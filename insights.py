"""
insights.py
-----------
Business logic that does not need scikit-learn: risk bands and retention advice.
"""
from typing import Dict, List

import config


def risk_band(probability: float) -> str:
    """Convert a churn probability into Low / Medium / High."""
    low, medium = config.RISK_THRESHOLDS
    if probability <= low:
        return config.RISK_LABELS[0]
    if probability <= medium:
        return config.RISK_LABELS[1]
    return config.RISK_LABELS[2]


def recommendations(customer: Dict, probability: float) -> List[str]:
    """Rule-based retention actions that mirror the findings of the analysis."""
    tips: List[str] = []
    if customer.get("Contract") == "Month-to-month":
        tips.append("Offer a discount or perk for moving to a one- or two-year contract.")
    if customer.get("InternetService") == "Fiber optic":
        tips.append("Check service quality and pricing complaints for this fiber customer.")
    if customer.get("PaymentMethod") == "Electronic check":
        tips.append("Encourage automatic payment (bank transfer / credit card) with a small incentive.")
    if float(customer.get("tenure", 99)) <= 12:
        tips.append("Customer is in the first year: schedule an onboarding check-in call.")
    if customer.get("InternetService") != "No" and customer.get("TechSupport") == "No":
        tips.append("Consider a free trial of Tech Support / Online Security (validate with an A/B test).")
    if float(customer.get("MonthlyCharges", 0)) > 80:
        tips.append("Bill is high: review the plan and offer a cheaper bundle if suitable.")
    if not tips or probability <= config.RISK_THRESHOLDS[0]:
        tips.append("Low risk: keep regular engagement; no special retention action needed.")
    return tips
