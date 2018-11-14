import app.lib.metrics
from app.config import logger

logger.info("Started application...")
app.lib.metrics.main()
