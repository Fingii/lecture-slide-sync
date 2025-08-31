import json
import logging.config
from pathlib import Path

logger = logging.getLogger("auto_slide_tracker")


def configure_loggers():
    config_path = Path(__file__).parent / "logging_config.json"
    with open(config_path) as f:
        config = json.load(f)

    for handler in config["handlers"].values():
        if "filename" in handler:
            log_path = Path(handler["filename"])
            log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.config.dictConfig(config)
