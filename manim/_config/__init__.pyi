from contextlib import _GeneratorContextManager
from typing import Any, Union

from rich import Console

from .utils import ManimConfig

logger: Any
console: Console
error_console: Console
config: Any
frame: Any

def tempconfig(temp: Union[ManimConfig, dict]) -> _GeneratorContextManager: ...
