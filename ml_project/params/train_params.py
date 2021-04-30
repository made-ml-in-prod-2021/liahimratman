from dataclasses import dataclass, field


@dataclass()
class TrainingParams:
    model_type: str = field(default="LinearRegression")
    max_iter: int = field(default=1000)
    random_state: int = field(default=100)
