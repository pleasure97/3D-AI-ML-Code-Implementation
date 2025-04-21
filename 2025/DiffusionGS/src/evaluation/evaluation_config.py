from dataclasses import dataclass
from pathlib import Path

@dataclass
class MethodConfig:
    name: str
    key: str
    path: Path

@dataclass
class EvaluationConfig:
    methods: list[MethodConfig]