from dataclasses import dataclass, field
from typing import Optional


@dataclass()
class TrainingParams:
    max_iter: Optional[int]
    random_state: Optional[int]
    model_type: str = field(default="LinearRegression")
