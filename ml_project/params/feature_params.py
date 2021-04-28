from dataclasses import dataclass
from typing import List, Optional


@dataclass()
class FeatureParams:
    features: List[str]
    target_col: Optional[str]
