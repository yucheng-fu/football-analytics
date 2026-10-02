from pydantic import BaseModel


class FeatureContribution(BaseModel):
    feature: str
    value: float | str | None
    contribution: float


class InferenceResponse(BaseModel):
    prediction: int
    probability: float | None = None
    timestamp: str
    model: str
    base_value: float
    contributions: list[FeatureContribution]
