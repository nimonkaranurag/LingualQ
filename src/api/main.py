from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from api.endpoints import translation, audio, quality, email
from contextlib import asynccontextmanager
from core.config import settings
from core.logging import setup_logging

import redis.asyncio as redis

from services.audio_processor import AudioProcessingService
from services.quality_checker import QualityChecker
from services.llm_interface import LLMService
from services.task_orchestrator import TaskOrchestrator, TaskType
from services.translations_memory_manager import TranslationsMemoryManager

audio_service = None
quality_checker = None
translations_memory_manager = None
task_orchestrator = None
llm_service = None
redis_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
	"""application life-span manager."""
	# define variables at the module level with the global keyword
	global audio_service, task_orchestrator, llm_service, quality_checker, redis_client, translations_memory_manager

	setup_logging()
	try:
		redis_client = redis.Redis(
			host=settings.REDIS_HOST,
			port=settings.REDIS_PORT,
			decode_responses=True
		)
		audio_service = AudioProcessingService()
		quality_checker = QualityChecker()
		llm_service = LLMService()
		translations_memory_manager = TranslationsMemoryManager()

		task_orchestrator = TaskOrchestrator(
			audio_service=audio_service,
			quality_checker=quality_checker,
			llm_service=llm_service,
			translations_memory_manager=translations_memory_manager
		)
		print("all services initialized successfully.")
		yield
	except Exception as e:
		print(f"Error during initialization: {str(e)}")
		raise
	finally:
		# Cleanup
		if redis_client:
			await redis_client.close()
		print("Cleanup Completed")

app = FastAPI(
	title="LingualQ",
	description="Enterprise Translation Memory Management System with AI-driven QA",
	version="1.0.0"
	)

# Configure server to accept cross-origin requests by setting CORS Headers
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
	)

# Include Routers
app.include_router(
	audio.router,
	prefix=f"{settings.API_V1_STR}/audio",
	tags=["audio"],
	dependencies=[Depends(get_audio_service)]
)
# a router object is an instance of fastapi.APIRouter

app.include_router(
	quality.router,
	prefix=f"{settings.API_V1_STR}/quality",
	tags=["quality"],
	dependencies=[Depends(get_quality_checker)]
)
app.include_router(
	translation.router,
	prefix=f"{settings.API_V1_STR}/translation",
	tags=["translation"],
	dependencies=[Depends(get_translations_memory_manager)]
)
app.include_router(
	email.router,
	prefix=f"{settings.API_V1_STR}/email",
	tags=["email"],
	dependencies=[Depends(get_llm_service)]
)

# route dependencies
async def get_task_orchestrator():
	return task_orchestrator
async def get_quality_checker():
	return quality_checker
async def get_llm_service():
	return llm_service
async def get_translation_memory_manager():
	return translations_memory_manager

@app.get("/health")
async def isHealthy():
	try:
		await redis_client.ping()
		service_status = {
			"redis": "connected",
			"audio_service": "initialized" if audio_service else "not initialized",
			"quality_service": "initialized" if quality_service else "not initialized",
			"llm_service": "initialized" if llm_service else "not initialized",
			"translations_memory_manager": "initialized" if translations_memory_manager else "not initialized",
			"task_orchestrator": "initialized" if task_orchestrator else "not initialized"
		}
		return {
			"status": "healthy",
			"services": services_status
		}
	except redis.RedisError:
		raise HTTPException(status_code=503,
			detail=f"Redis Connection Failed.")

	except Exception as e:
		raise HTTPException(status_code=500,
			detail=f"Health Check Failed: {str(e)}")

@app.get(f"task/{task_id}")
async def get_task_status(
	task_id: str,
	task_orchestrator: TaskOrchestrator = Depends(task_orchestrator)
):
	"""Get task status of any task by its task_id."""
	try:
		task = task_orchestrator.task_queue.get(task_id)
		if not task:
			raise HTTPException(
				status_code=404,
				detail="Task not found"
			)
		return {
			"task_id": task_id,
			"status": task["status"],
			"result": task.get("result"),
			"error": task.get("error")
		}
	except Exception as e:
		raise HTTPException(
			status_code=400,
			detail=str(e)
		)

if __name__=="__main__":
	import uvicorn

	uvicorn.run(
		"main.app",
		host=settings.HOST,
		port=settings.PORT,
		reload=settings.DEBUG,
		workers=settings.WORKERS
	)

