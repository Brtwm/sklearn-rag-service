from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

class PredictRequest(BaseModel):
    """13 химических показателей одной партии вина."""
    model_config = ConfigDict(populate_by_name=True)
    
    alcohol: float = Field(..., ge=0, description="Содержание алкоголя, %")
    malic_acid: float = Field(..., ge=0, description="Малоновая кислота, г/л")
    ash: float = Field(..., ge=0, description="Зольность")
    alcalinity_of_ash: float = Field(..., ge=0, description="Щёлочность золы")
    magnesium: float = Field(..., ge=0, description="Магний, мг/л")
    total_phenols: float = Field(..., ge=0, description="Общие фенолы")
    flavanoids: float = Field(..., ge=0, description="Флавоноиды")
    nonflavanoid_phenols: float = Field(..., ge=0, description="Нефлавоноидные фенолы")
    proanthocyanins: float = Field(..., ge=0, description="Проантоцианидины")
    color_intensity: float = Field(..., ge=0, description="Цветовая интенсивность")
    hue: float = Field(..., ge=0, description="Оттенок")
    od280_od315_of_diluted_wines: float = Field(
        ..., ge=0, alias="od280/od315_of_diluted_wines", description="OD280/OD315"
    )
    proline: float = Field(..., ge=0, description="Пролин, мг/л")

class PredictResponse(BaseModel):
    """Результат предсказания модели."""

    predicted_class: Literal["class_0", "class_1", "class_2"]
    probabilities: dict[str, float] = Field(
        ..., description="Вероятность каждого класса (сумма ≈ 1.0)"
    )