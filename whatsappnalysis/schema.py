from dataclasses import dataclass
from enum import EnumMeta
from typing import Dict


@dataclass
class Schema:
    columns: EnumMeta
    columns_to_dtypes: Dict[str, type]
