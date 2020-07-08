import logging
from rich.logging import RichHandler
from rich.console import Console

console=Console(record=True)

logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console)]
)

logger = logging.getLogger("rich")
