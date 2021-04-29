from dataclasses import dataclass, field


@dataclass()
class ScalerParams:
    mean: str = field(default=0)
    scale: int = field(default=1)
