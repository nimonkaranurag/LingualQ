from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from enum import Enum

class TaskStatus(str, Enum):
	PENDING = "pending"
	IN_PROGRESS = "in_progress"
	COMPLETED = "completed"
	FAILED = "failed"

class AudioTranscriptionRequest(BaseModel):
	language: Optional[str] = Field(None, description="Source lang code.")
	model_type: Optional[str] = Field("default", description="Transcription model.")
	enhance_audio: Optional[bool] = Field(False, description="Apply audio enhancement.")

class TaskResponse(BaseModel):
	task_id: str
	status: str

class AudioProcessingRequest(BaseModel):
	status: str
	result: Optional[Dict] = None
	error: Optional[str] = None

class TranslationQualityRequest(BaseModel):
	source_text: str
	translated_text: str
	source_lang: str
	target_lang: str

class QualityCheckRequest(BaseModel):
	text: str
	content_type: Optional[str] = Field("general", description="Type of content being checked")

class QualityScores(BaseModel):
	overall_score: float = Field(..., ge=0.0, le=1.0)
	fluency_score: Optional[float] = Field(None, ge=0.0, le=1.0)
	grammar_score: Optional[float] = Field(None, ge=0.0, le=1.0)
	semantic_score: Optional[float] = Field(None, ge=0.0, le=1.0)
	completeness_score: Optional[float] = Field(None, ge=0.0, le=1.0)

class QualityCheckResponse(BaseModel):
	scores: QualityScores
	error_analysis: Dict[str, List[str]]
	suggestions: List[str]

class EmailGenerationRequest(BaseModel):
	meeting_summary: str
	action_items: List[Dict[str, str]]
	email_type: Optional[str] = "follow_up"

class EmailGenerationResponse(BaseModel):
	subject: str
	body: str
	suggested_recipients: List[str]
	quality_check: QualityCheckResponse

class TranslationRequest(BaseModel):
	source_text: str = Field(...., description="Text to be translated")
	source_lang: str = Field(..., description="Source language code")
	target_lang: str = Field(..., description="Target language code")
	use_memory: bool = Field(True, description="Whether to use translation memory")
	quality_check: bool = Field(True, description="Whether to perform quality check")

class TranslationResponse(BaseModel):
	translated_text: str
	quality_scores: Optional['QualityScores'] = None
	from_memory: bool = False
	similarity_score: Optional[float] = None
