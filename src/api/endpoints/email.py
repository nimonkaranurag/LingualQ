from fastapi import APIRouter, HTTPException
from api.models.schemas import (
	TaskStatus,
	TaskResponse,
	EmailGenerationRequest,
	EmailResponse
)
from services.task_orchestrator import TaskType
router = APIRouter()

@router.post("/generate", reponse_model=TaskResponse)
async def generate_email(request: EmailGenerationRequest):
	"""Generate email using meeting summary."""
	try:
		task_id = task_orchestrator.create_task(
			TaskType.EMAIL_GENERATION,
			{
				"summary": request.meeting_summary,
				"action_items": request.action_items,
				"email_type": request.email_type
			})
		return {
			"task_id": task_id,
			"status": TaskStatus.PENDING
		}
	except exception as e:
		raise HTTPException(status_code=400, detail=str(e))
