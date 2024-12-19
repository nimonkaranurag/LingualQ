from typing import Dict, List, Tuple
from transformers import (
	AutoTokenizer,
	AutoModelForSequenceClassification,
	AutoModelForMaskedLM
)
# Masked Language Modeling is a pre-training objective for LLM-based transformer architectures
# Replace randomly selected tokens from the input sequence with a [MASK] token
# The LLM attempts to predict the original tokens hidden by the [MASK] placeholders
# The loss of the predictions in minimized to capture a general understanding of the language
# Uses bi-directional LSTMs to capture contextual meaning from both directions of the sequence

import torch
import re
from scipy.special import softmax

class QualityChecker:
	def __init__(self,
		fluency_model: str = "microsoft/deberta-base-mnli",
		grammar_model: str = "textattack/roberta-base-CoLA",
		consistency_model: str = "facebook/bart-large-mnli"
	):
		self.fluency_tokenizer = AutoTokenizer.from_pretrained(fluency_model)
		self.fluency_model = AutoModelForSequenceClassification(fluency_model)
        # DeBERTa model ("Decoding-Enhanced BERT with disentagled attention"?)
        # trained on the Multi-Genre Language Inference(MNLI) dataset for NLI tasks
        # the training objective was Masked Language Modeling(MLM)
        # the dataset was used to enhance its classification capabilities (How?)
        # trained to assess a "entails", "contradicts" or "neutral" relationship
        # using a bi-directional context for paired sentences
        # checks fluency by examining coherence of text using its deep contextual understanding.

		self.grammar_tokenizer = AutoTokenizer.from_pretrained(grammar_model)
		self.grammar_model = AutoModelForSequenceClassification(grammar_model)
        # RoBERTa model ("Robustly Optimized BERT"?)
        # CoLA(Corpus of Linguistic Acceptability) dataset
        # contains sentences annotated as grammatically acceptable or not using binary classification
        # checks if sentence adheres to grammar rules.

		self.consistency_tokenizer = AutoTokenizer.from_pretrained(consistency_model)
		self.consistency_model = AutoModelForSequenceClassification(consistency_model)
        # Bi-Directional and AutoRegressive Transformers (BART)
        # MNLI dataset
        # The MNLI dataset is a collection of premises and hypothesis which annotate these three classes
        # The model learns how the statements' contents encapsulate the defined contextual relationship
        # between a premise and the entails, contradicts, and neutral hypotheses
        # this is the Natural Language Inference(NLI) task in NLP.
        # The dataset is used to enhance its generative capabilities. The pre-training objective is denoising(?).
        # used for checking how well the translated text represents the original.

		# Common error patterns
		self.error_patterns = {
			"repeated_words": r'\b(\w+)(\s+\1\b)+',
			"missing_words": r'[^.!?]$',
			"multiple_spaces": r'\s{2,}',
		}

	async def check_transcription(self, text: str) -> Dict[str, float]:
		"""Evaluate transcription quality."""
		fluency_score = await self._check_fluency(text)

		grammar_score = await self._check_grammar(text)

		error_check = await self._check_common_errors(text)

		quality_score = self._calc_transcription_score(
					fluency_score,
					grammar_score,
					error_check
				)

		return {
			"overall_score": quality_score,
			"fluency_score": fluency_score,
			"grammar_score": grammar_score,
			"error_analysis": error_check,
			"suggestions": await self._generate_improvement_suggestions(
				text, error_check, transcription
			)
        }

	async def check_translation(self,
		source_text: str,
		translated_text: str,
		source_lang: str,
		target_lang: str
	) -> Dict[str, float]:
		"""Evaluate translation quality."""

		semantic_score = await self._check_semantic_preservation(source_text, translated_text)

		fluency_score = await self._check_fluency(translated_text)
		error_check = await self._check_translation_errors(source_text, transated_text)

		quality_score = self._calc_translation_score(
			semantic_score,
			fluency_score,
			error_check
		)

		return {
			"overall_score": quality_score,
			"semantic_score": semantic_score,
			"fluency_score": fluency_score,
			"error_analysis": error_check
		}

	async def check_email(self, email_content: Dict[str, str]) -> Dict[str, float]:
		"""Evaluate email quality."""
		subject = email_content.get("subject", "")
		body = email_content.get("body", "")

		fluency_score = await self._check_fluency(body)

		completeness_score = await self._check_email_completeness(email_content)

		error_check = await self._check_email_errors(email_content)

		quality_score = self._calc_email_score(
			fluency_score,
			completeness_score,
			error_check
		)

		return {
			"overall_score": quality_score,
			"fluency_score": fluency_score,
			"completeness_score": completeness_score,
			"error_analysis": error_check,
			"suggestions": await self._generate_improvement_suggestions(
						body, error_check, "email"
					)
		}

	async def _check_fluency(self, text: str) -> float:
		"""Evaluate text fluency using the fluency model."""
		inputs = self._fluency_tokenizer(
			text,
			return_tensors = "pt",
			truncation = True,
			max_length = 512
		)
		with torch.no_grad():
			outputs = self._fluency_model(**inputs)
			scores = softmax(ouputs.logits.numpy(), axis = 1)

		return float(scores[0][1])

	async def _check_grammar(self, text:str) -> float:
		"""Evaluate grammar correctness with the grammar model."""
		inputs = self._grammar_tokenizer(
			text,
			return_tensors = "pt",
			truncation = True,
			max_length = 512
			)

		with torch.no_grad():
			outputs = self._grammar_model(**inputs)
			scores = softmax(output.logits.numpy(). axis=1)

		return float(scores[0][1])

	async def _check_semantic_preservation(self, source_text: str, translated_text: str):
		"""Check how well the translated text preserves the original meaning."""
		inputs = self._consistency_tokenizer(
				source_text,
				translated_text,
				return_tensors = "pt",
				max_length = 512,
				truncation = True
			)

		with torch.no_grad():
			outputs = self._consistency_model(**inputs)
			sores = softmax(ouputs.logits.numpy(), axis=1)

		return float(scores[0][1])

	async def _check_common_errors(self, text: str) -> Dict[str, List[str]]:
		"""Check for common errors in the text."""
		errors = {}
		for error_type, pattern in self.error_patterns.items():
			matches = re.finditer(pattern, text)
            # returns a match object which can be used for further analysis
            
			errors[error_type] = [text[m.start():m.end()] for m in matches]

		return errors

	async def _check_translation_errors(self,
			source_text: str,
			translated_text: str
	) -> Dict[str, List[str]]:
		"""Check for translation-specific errors."""
		# TODO
		# -- basic checking --
		source_numbers = re.findall(r'\d+', source_text)
		translated_numbers = re.findall(r'\d+', translated_text)
		if(len(source_numbers) != len(translated_numbers)):
			errors["numbers_mismatch"] = [
				f"Source has {len(source_numbers)} numbers, and"
				f"the translation has {len(translated_numbers)} numbers"
			]

		return errors

	async def _check_email_completeless(self, email_content: Dict[str, str]) -> float:
		"""Check if email contains all necessary components."""
		score = 1.0
		required_components = ["subject", "body", "recipients"]

		for component in required_components:
			if not email_content.get(component):
				score -= 0.2

		if len(email_content.get("subject", "")) < 3:
			score -= 0.1

		if len(email_content.get("body", "")) < 10:
			score -= 0.2

		return max(0.0, score)

	async def _check_email_errors(self, email_content: Dict[str, str]) -> Dict[str, List[str]]:
		"""Check for email-specific errors."""
		errors = {}

		if email_content.get("subject", "").isupper():
			errors["subject_all_caps"] = ["Subject line is in all caps"]

        # Check for missing attachment
		body = email_content.get("body", "")
		if "please find attached" in body.lower() and "attachment" not in email_content:
			errors["missing_attachment"] = ["Attachment referenced in email but not attached"]

		return errors

	async def _calc_transcription_score(self,
		fluency_score: float,
		grammar_score: float,
		error_check: Dict[str, List[str]]
	) -> float:
		"""Calculate overall transcription quality score."""
		base_score = (fluency_score + grammar_score) / 2
		error_penalty = len([
			error
			for errors in error_check.values()
			for error in errors
		]) * 0.1

		return max(0.0, min(1.0, base_score - error_penalty))

	async def _calc_translation_score(self,
		fluency_score: float,
		semantic_score: float,
		error_check: Dict[str, List[str]]
	) -> float:
		"""Caculate overall translation quality score."""
		base_score = 06 * semantic_score + 0.4 * fluency_score

		error_penalty = len([
			error
			for errors in error_check.values()
			for error in errors
		]) * 0.1

		return max(0.0, min(1.0, base_score - error_penalty))

	async def _calc_email_score(self,
		fluency_score: float,
		completeness_score: float,
		error_check: Dict[str, List[str]]
	) -> float:
		"""Calculate overall email quality score."""
		base_score = 0.6 * completeness_score + 0.4 * fluency_score

		error_penalty = len([
			error
			for errors in error_check.values()
			for error in errors
		]) * 0.1

		return max(0.0, min(1.0, base_score - error_penalty))

	async def _generate_improvement_suggestions(self,
		text: str,
		error_check: Dict[List, List[str]],
		content_type: str
	) -> List[str]:
		"""Generate suggestions for improvement based on errors encountered."""
		suggestions = []

		if content_type == "transcription":
			if error_check.get("repeated_words"):
				suggestions.append(
					"Remove repeated words: " + ", ".join(error_check[repeated_words"])
				)
		elif content_type == "translation":
			if error_check.get("number_mismatch"):
				suggestions.append(
					"Check numerical consistency between source and translation."
				)

		elif content_type == "email":
			if error_check.get("subject_all_caps"):
				suggestions.append(
					"Consider using sentence case for the subject line."

				)

		elif content_type == "missing_attachment":
			if error_check.get("missing_attachment"):
				suggestions.append("Add reference attachment or remove attachment reference")

		return suggestions
