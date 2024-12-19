from fastapi import APIRouter, HTTPException
from api.models.schemas import (
	QualityCheckRequest,
	QualityCheckResponse,
	TranslationQualityRequest
)

router = APIRouter()

@router.post("/check/translation", response_model=QualityCheckResponse)
async def check_translation_quality(request: TranslationQualityRequest):
	"""Check the quality of a translated pair(source and target)."""
	try:
		quality_result = await quality_checker.check_translation(
			source_text = request.source_text
			translated_text = request.translated_text
			source_lang = request.source_lang
			target_lang = request.target_lang
		)
		return quality_result
	except Exception as e:
		raise HTTPException(status_code=400,detail=str(e))

@router.post("/check/transcription", response_model=QualityCheckResponse)
async def check_transcription_quality(request: QualityCheckRequest):
	"""Check the quality of a transcription."""
	try:
		quality_result = await quality_checker.check_transcription(
			text = request.text
		)
		return quality_result
	except Exception as e:
		raise HTTPException(status_code=400,detail=str(e))

@router.post("/check/email", reponse_model=QualityCheckResponse)
async def check_email_quality(request: QualityCheckRequest):
	"""Check the quality of a generated email."""
	try:
		quality_result = await quality_checker.check_email_quality(
			email_content = request
		)
		return quality_result
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))
