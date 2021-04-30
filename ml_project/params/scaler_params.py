from dataclasses import dataclass
from typing import List


@dataclass()
class ScalerParams:
    mean: List[str]
    scale: List[str]
