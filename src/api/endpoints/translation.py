from fastapi import APIRouter, HTTPException
from api.models.schema import (
	TranslationRequest,
	TranslationResponse,
	TaskResponse,
	TaskStatus
)
from services.task_orchestrator import TaskType

router = APIRouter()
@router.post("/translate", response_model=TaskResponse)
async def translate_text(request: TranslationRequest):
	"""Translate text."""
	try:
		task_id = await task_orchestrator.create_task(
			TaskType.TRANSLATION,
			{
				"text": request.text,
				"source_lang": request.source_lang,
				"target_lang": request.target_lang
			})
		return {
			"task_id": task_id,
			"status": TaskStatus.PENDING
		}
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

@router.post("/summarize", response_model=TaskResponse)
async def summarize_text(text: str):
	"""Summarize text(used internally, by the app)."""
	try:
		task_id = await task_orchestrator.create_task(
			TaskType.SUMMARIZE,
			{"text": text})
		return {
			"task_id": task_id,
			"status": TaskStatus.PENDING
		}
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))
