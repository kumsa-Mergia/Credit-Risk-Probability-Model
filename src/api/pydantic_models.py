from pydantic import BaseModel
from typing import Optional


class CustomerInput(BaseModel):
    Amount: float
    Value: float
    CountryCode: int
    PricingStrategy: int
    transaction_hour: int
    transaction_day_of_week: int
    transaction_day_of_month: int
    transaction_month: int
    transaction_year: int
    amount_sum_accountid: float
    amount_mean_accountid: float
    amount_std_accountid: float
    value_sum_accountid: float
    value_mean_accountid: float
    value_std_accountid: float
    transaction_count_accountid: int
    CurrencyCode_KES: Optional[int] = 0
    CurrencyCode_USD: Optional[int] = 0
    ProductCategory_Other: Optional[int] = 0
    ChannelId_Web: Optional[int] = 0


class RiskPrediction(BaseModel):
    risk_probability: float
