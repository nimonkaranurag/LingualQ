from loguru import logger
import logging
import sys

class InterceptHandler(logging.Handler):
	def emit(self, record: logging.LogRecord) -> None:
		"""Emits a log record."""
		try:
			level = logger.level(record.levelname).name
		except ValueError:
			level = record.levelno

		frame, depth = logging.currentframe(), 2
		while frame.f_code.co_filename == logging.__file__:
			frame = frame.f_back
			depth += 1
		logger.opt(depth=depth, exception=record.exc_info).log(
			level, record.getMessage()
		)

class JSONFormatter:
	def __init__(self):
		self.format_dict = {
			"timestamp": "%(asctime)s",
			"level": "%(levelname)s",
			"message": "%(message)s",
			"module": "%(module)s"
		}

	def format(self, record: logging.LogRecord) -> str:
		log_dict = {}
		for key, value in self.format_dict.items():
			log_dict[key] = value % record.__dict__
		if record.exc_info:
			log_dict["exc_info"] = self.formatException(record.exc_info)
		return json.dumps(log_dict)
	def format_exception(self, exc_info) -> str:
		return logging.Formatter().formatException(exc_info)

def configure_logging() -> None:
	"""Set-up logging configuration."""
	logging.root.handlers = [] # clear all existing handlers

	# Set-up loguru
	loguru.configure(
		handlers=[
			{
				"sink": sys.stdout,
				"format":"<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                         		"<level>{level: <8}</level> | "
                         		"<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                         		"<level>{message}</level>", 
				"level": settings.LOG_LEVEL
			}
		]
	)
	# Intercept standard library logging.
	logging.basicConfig(handlers=[InterceptHandler()], level=0)

	# Set logging levels for specific modules
	for logger_name in [
		"uvicorn",
		"uvicorn.error",
		"fastapi",
		"huggingface",
		"torch"
	]:
		logging.getLogger(logger_name).handlers = [InterceptHandler()]
