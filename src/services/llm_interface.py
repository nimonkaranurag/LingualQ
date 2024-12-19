from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqGeneration
import torch
from typing import Dict, List

class LLMService:
	def __init__(self,
		summarization_model: str = "facebook/bart-large-cnn",
		generation_model: str = "gpt2-medium",
		translation_model: str = "Helsinki-NLP/opus-mt-en-ROMANCE"
	):
		self.summarizer = pipeline(
			"summarization",
			model = summarization_model,
			device = 0 if torch.cuda.is_available() else -1
		)

		self.generator_tokenizer = AutoTokenizer.from_pretrained(generation_model)
		self.generator_model = AutoModelForCausalLM.from_pretrained(generation_model)
        # used for text generation tasks

		self.translation_tokenizer = AutoTokenizer.from_pretrained(translation_model)
		self.translation_model = AutoModelForSeq2SeqGeneration.from_pretrained(translation_model)

	async def summarize(self, text: str, max_length: int = 130) -> str:
		"""Generate a concise summary of input text."""
		summary: List[Dict[str,str]] = self.summarizer(
				text = text,
				max_length = max_length,
				min_length = 30,
				do_sample = False
			)
		return summary[0]["summary_text"]

	async def extract_key_points(self, text: str) -> List[str]:
		"""Extract key discussion points from text."""
		prompt = f"Please reference the following meeting summary:\n\n{text}\n\nKey Points/Highlights:"

		input_data = self.generator_tokenizer(
					prompt,
					return_tensors = "pt",
					max_length = 1024,
					truncation = True
				)

		outputs = self.generator_model.generate(
					input_data.input_ids,
					max_length = 150,
					num_return_sequences = 1,
					temperature = 0.7,
					top_p = 0.9
				)

		key_points = self.generator_tokenizer.decode(outputs[0])
		# Process and clean the output to return a list of key points
		key_points = [
					point.strip()
					for point in key_points.split('\n')
					if point.strip() and point.strip().startswith('-')
				]

		return key_points

	async def extract_action_items(self, text: str) -> List[Dict]:
		"""Extract action items with asignees and deadlines."""
		prompt = f"Action items from the following text:\n\n{text}\n\nare as follows:"

		input_data = self.generator_tokenizer(
					prompt,
					return_tensors="pt",
					max_length=1024,
					truncation=True
				)
		outputs = self.generator_model.generate(
					input_data.input_ids,
					max_length=150,
					num_return_sequences=1,
					temperature=0.7
				)

		action_text = self.generator_tokenizer.decode(outputs[0])
		# Process the output in a structured format.
		# -- requires more robust parsing --
		actions = []
		for line in action_text.split('\n'):
			if ':' in line:
				task, assignee = line.split(':')
				actions.append({
					"task": task.strip(),
					"assignee": assignee.strip(),
					"deadline": None # -- extract from text --
				})

		return actions

	async def generate_email(self, summary: str, action_items: List[Dict], email_type: str = "follow_up") -> Dict:
		"""Generate a follow-up email based on the meeting's summary and actionables."""
		action_items_text = "\n".join(
					f"- {item['task']} (Assignee: {item['assignee']})"
					for item in action_items
					)

		prompt = f"""
		Email_type: {email_type},
		Meeting_Summary: {summary},
		Action_Items: {action_items_text}

		Generate a professional email with a subject and body.
		"""

		inputs = self.generator_tokenizer(
			prompt,
			return_tensors="pt",
			max_length=1024,
			truncation=True
			)
		outputs = self.generator_model.generate(
			inputs.input_ids,
			max_length=500,
			num_return_sequences=1,
			temperature=0.7
			)

		email_text = self.generator_tokenizer.decode(outputs[0])

		# Parse generated email into subject and body.
		# -- oversimplified --
		lines = email_text.split('\n')
		subject = lines[0].replace("Subject: ", "").strip()
		body = '\n'.join(lines[2:]).strip()

		return {
			"subject": subject,
			"body": body,
			"suggested_recepients": [item["assignee"] for item in action_items]
			}

	async def translate(self, text: str, source_lang: str, target_lang: str) -> str:
		"""Translate text between languages."""
		inputs = self.translation_tokenizer(
				text,
				return_tensors="pt",
				padding=True,
				truncation=True
				)
		outputs = self.translation_model.generate(
				inputs.input_ids,
				max_length = 1024,
				num_beams = 4,
				length_penalty = 0.6
				)
		translation = self.translation_tokenizer.decode(outputs[0], skip_special_tokens = True)

		return translation

