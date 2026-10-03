from pydantic import BaseModel, Field


class Prediction(BaseModel):
    class_name: str = Field(alias="class")
    confidence: float


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    top_predictions: list[Prediction]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    classes: int
