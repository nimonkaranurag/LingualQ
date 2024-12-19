from pydantic import BaseSettings

class Settings(BaseSettings):
	# API Settings
	API_V1_STR: str = "/api/v1"
	PROJECT_NAME: str = "LingualQ"
	HOST: str = "0.0.0.0"
	PORT: int = 8000
	DEBUG: bool = True

	# Redis Settings
	REDIS_HOST: str = "localhost"
	REDIS_PORT: int = 6379
	REDIS_DB: int = 0
	REDIS_PASSWORD: str = None

	# CORS Settings
	BACKEND_CORS_ORIGINS: list = ["*"]

	# Model Settings
	AUDIO_MODEL: str = "facebook/wav2vec2-base-960h"
	TRANSLATION_MODEL: str = "Helsinki-NLP/opus-mt-en-ROMANCE"
	QUALITY_CHECK_MODEL: str = "microsoft/deberta-base-mnli"

	# Processing Settings
	MAX_AUDIO_LENGTH: int = 600 # seconds
	MAX_TEXT_LENGTH: int = 5000 # characters
	BATCH_SIZE: int = 32

	# Infrastructure Settings
	USE_CUDA: bool = True
	WORKERS: int = 4

	# Logging settings
	LOG_LEVEL: str = "INFO"
	LOG_FORMAT: str = f"%(asctime)s  - %(name)s - %(levelname)s - %(message)s"

	# Task Queue Settings
	TASK_QUEUE_TIMEOUT: int = 3600
	MAX_RETIRES: int = 3

	class Config:
		case_sensitive = True
		env_file = ".env"

settings = Settings()
