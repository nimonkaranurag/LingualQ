from enum import Enum
import asyncio
from datetime import datetime
from typing import List, Dict

class TaskType(Enum):
	TRANSCRIPTION = "transcription"
	TRANSLATION = "translation"
	SUMMARIZATION = "summarization"
	EMAIL_GENERATION = "email_generation"

class TaskStatus(Enum):
	PENDING = "pending"
	IN_PROGRESS = "in_progress"
	COMPLETED = "completed"
	FAILED = "failed"

class Task_Orchestrator:
	def __init__(
		self, audio_service, translation_service,
		llm_service, quality_service
	):
		self.audio_service = audio_service
		self.translation_service = translation_service
		self.llm_service = llm_service
		self.quality_service = quality_service
		self.task_queue: Dict[str, Dict] = {}

	async def create_task(
		self,
		task_type: TaskType,
		input_data: Dict,
		priority: int = 1
	) -> str:
		task_id = f"{task_type.value}_{datetime.utcnow().timestamp()}"
		self.task_queue[task_id] = {
			"type": task_type,
			"status": TaskStatus.PENDING,
			"input": input_data,
			"priority": priority,
			"created_at": datetime.utcnow(),
			"result": None,
			"error": None
		}

		asyncio.create_task(self._process_task(task_id))
		return task_id

	async def _process_task(self, task_id: str):
		task = self.task_queue[task_id]
		task["status"] = TaskStatus.IN_PROGRESS

		try:
			if task["type"] == TaskType.TRANSCRIPTION:
				result = await self._transcription_handler(task["input"])
			elif task["type"] == TaskType.TRANSLATION:
				result = await self._translation_handler(task["input"])
			elif task["type"] == TaskType.SUMMARIZATION:
				result = await self._summarization_handler(task["input"])
			elif task["type"] == TaskType.EMAIL_GENERATION:
				result = await self._email_gen_handler(task["input"])

			task["result"] = result
			task["status"] = TaskStatus.COMPLETED
		except Exception as e:
			task.STATUS = TaskStatus.FAILED
			task["error"] = str(e)

	async def _transcription_handler(self, input_data: Dict) -> Dict:
		"""Handles audio transcription with quality checks"""
		audio_data = input_data["audio"]

		transcription = await self.audio_service.process_audio(audio_data)
		quality_score = await self.quality_service.check_transcription(
						transcription["transcription"]
					)
		return {
			"transcription": transcription["transcription"],
			"confidence": transcription["confidence_score"],
			"quality_score": quality_score
		}

	async def _translation_handler(self, input_data: Dict) -> Dict:
		"""Handles text translation with memory look-up and quality assurance."""

		text = input_data["text"]
		source_lang = input_data["source_lang"]
		target_lang = input_data["target_lang"]

		# Check translations database
		similar_translations = self.translation_service.find_similar_translations(
						text, source_lang, target_lang
					)

		if similar_translations and similar_translations[0]["similarity"] > 0.95:
			return similar_translations[0]

		# Generate a new translation
		translation = await self.llm_service.translate(
				text, source_lang, target_lang
				)

		quality_score = await self.quality_service.check_translation(
				text, translation, source_lang, target_lang
				)

		return {
			"translation": translation,
			"quality_score": quality_score
			}

	async def _summarization_handler(self, input_data: Dict) -> Dict:
		"""Generate meeting summary with key point extraction."""
		text = input_data["text"]

		# Generate summary
		summary = await self.llm_service.summarize(text)

		# Extract key points and action items
		key_points = await self.llm_service.extract_key_points(text)
		action_items = await self.llm_service.extract_action_items(text)

		return {
			"summary": summary,
			"key_points": key_points,
			"action_items": action_items
			}

	async def _email_gen_handler(self, input_data: Dict) -> Dict:
		"""Generate follow-up email based on meeting summary."""
		summary = input_data["summary"]
		action_items = input_data["action_items"]
		email_type = input_data.get("email_type", "follow-up")

		email_content = self.llm_service.generate_email(
				summary = summary,
				action_items = action_items,
				email_type = email_type
				)

		return {
			"subject": email_content["subject"],
			"body": email_content["body"],
			"recepients": email_content.get("suggested_recepients",[])
			}
