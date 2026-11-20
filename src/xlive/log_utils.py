import logging
import os
from datetime import datetime


def setup_logging():
    """Configure the process-wide X-Live logger.

    A timestamped file is added only when ``XTALK_LOG_DIR`` is set.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    logs_dir = os.getenv("XTALK_LOG_DIR")
    if logs_dir:
        os.makedirs(logs_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        handlers.append(
            logging.FileHandler(
                os.path.join(logs_dir, f"xtalk_{timestamp}.log"), encoding="utf-8"
            )
        )

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )

    return logging.getLogger("xlive")


logger = setup_logging()
