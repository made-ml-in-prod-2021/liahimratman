from dataclasses import dataclass
from typing import List, Optional


@dataclass()
class ScalerParams:
    mean: str = List[str]
    scale: int = List[str]
