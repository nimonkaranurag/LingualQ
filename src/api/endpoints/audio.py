from typing import Dict
from fastapi import APIRouter, UploadFile, File, HTTPException
from api.models.schemas import (
	TaskResponse,
	AudioTranscriptionRequest,
	AudioProcessingResponse
)


router = APIRouter()

@router.post("/transcribe", response_model=TaskResponse)
async def transcribe_audio(
	background_tasks: BackgroundTasks,
	file: UploadFile = File(...),
	config: AudioTranscriptionRequest = None
):
	"""Transcribe audio file."""
	try:
		audio_content = await file.read()

		# Create a transcription task
		task_id = await task_orchestrator.create_task(
				TaskType.TRANSCRIPTION,
				{
					"audio": audio_content,
					"config": config.dict() if config else {}
				}
		)

		return {"task_id": task_id, "status": "pending"}
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

@router.get("/task/{task_id}", response_model=AudioProcessingResponse)
async def get_transcription_result(task_id: str):
	"""Get the result of an audio processing task."""
	try:
		task = task_orchestrator.task_queue.get(task_id)
		if not task:
			raise HTTPException(status_code=404, detail="Task not found")

		return {
			"status": task["status"].value()
			"result": task.get("result")
			"error": task.get("error")
		}
	except Exception as e:
		raise HTTPException(status_code=404, detail=str(e))
