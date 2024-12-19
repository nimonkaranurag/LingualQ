from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Dict, Optional

class TranslationsMemoryManager:
	def __init__(self, model_name: str = "xlm-roberta-base"):
		self.tokenizer = AutoTokenizer.from_pretrained(model_name)
		self.model = AutoModel.from_pretrained(model_name)
		self.translation_memory: Dict[str, Dict] = {}

	def compute_embeddings(self, text: str) -> np.ndarray:
		input = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
		with torch.no_grad():
			outputs = self.model(**inputs)
		return outputs.last_hidden_state.mean(dim=1).numpy()

	def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
		# Compute coscine similarity
		return float(np.dot(embedding1.flatten(),embedding2.flatten())/np.linalg.norm(embedding1)*np.linalg.norm(embedding2)))

	def find_similar_translations(self,
		source_text: str,
		source_lang: str,
		target_lang: str,
		threshold: float = 0.8) -> List[Dict]:
		# Find similar translations from buffer
		query_embedding = self.compute_embeddings(source_text)

		target_translations = []
		for key, entry in self.translation_memory.items():
            key_prefix = f"{source_lang}_{target_lang}_"
            if key.startswith(key_prefix):
			    if(entry["source_lang"]==source_lang and entry["target_lang"]==target_lang):
				    similarity = self.compute_similarity(query_embedding, entry["source_embeddings"])
				    if similarity >= threshold:
					    target_translations.append({
						    "source_text": entry["source_text"],
						    "target_text": entry["target_text"],
						    "similarity": float(similarity),
						    "quality_score": entry["quality_score"]})

		return sorted(target_translations, key=lambda x: x["similarity"], reverse=True)

	def add_translation(self,
		source_text: str,
		target_text: str,
		source_lang: str,
		target_lang: str,
		quality_score: float) -> Dict:

		# Add a new translation to the buffer.
		key = f"{source_lang}_{target_lang}_{len(translation_memory)}"

		self.translation_memory[key] = {
			"source_text": source_text,
			"target_text": target_text,
			"source_lang": source_lang,
			"target_lang": target_lang,
			"source_embeddings": compute_embeddings(source_text),
			"quality_score": quality_score
			}

		return {"id": key, "source_text": source_text, "target_text": target_text, "quality_score": quality_score}

