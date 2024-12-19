from typing import Dict
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
# Wav2Vec etymology: Waveform to Vector, generates meaningful embedding vectors for each raw audio sample
# much like NLP tasks, these embeddings capture features relevant to audio processing tasks

# CTC loss function is used for sequence-to-sequence(tasks like ASR) conversion tasks
# The Wav2Vec2Processor is for the pre and post processing pipelines
import librosa

class AudioProcessingService:
	def __init__(self,
		model_name: str = "facebook/wav2vec2-base-960h",
		sample_rate: int = 16000
	):
		self.processor = Wav2Vec2Processor.from_pretrained(model_name)
		self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        # lightweight model trained on LibriSpeech-960h dataset
        # usecase: audio transcription (for English audio)
		self.sample_rate = sample_rate

	async def process_audio(self, audio_file: bytes) -> Dict[str, str]:
		"""Process audio file and return transcription."""
		try:
			audio = self._load_audio(audio_file)
			inputs = self.processor(
				audio,
				sampling_rate = self.sample_rate,
				return_tensors = "pt",
				padding = True # padded to match longest sequence
			)
            # inputs.keys() = ["input_values","attention_mask"]
            # type(input_values) = PyTorch Tensor
            # inputs.input_values.shape = (batch_size, len(longest_sequence))
			with torch.no_grad():
				logits = self.model(inputs.input_values).logits
			predicted_ids = torch.argmax(logits, dim=-1)
            # logits.shape = (batch_size, len(longest_sequence), # of token_ids in model vocab)
            # each column in the logits tensor is a token_id from the model's vocabulary
            
            # predicted_ids.shape = (batch_size, len(longest_sequence)

            # thus, the original input shape is restored
            # however, unlike the inputs matrix where each entry was a discrete audio sample
            # in the predicted_ids tensor, each entry is a predicted token_id
            # torch.argmax(..) is used to return the index of the highest class score

			transcription = self.processor.batch_decode(predicted_ids)[0]
            # maps raw class scores to valid token_ids
            # transcription.shape = (batch_size,)
            # type(transcription) = List[str]
            # each item is the correspondingly translated String for each sequence in the predicted_ids tensor 

			return {
				"transcription": transcription,
				"confidence_score": self._calc_confidence_score(logits)
			}
		except Exception as e:
			raise RuntimeError(f"Error processing audio: {str(e)}")

	def _load_audio(self, audio_file: bytes) -> np.ndarray:
		"""Load and pre-process an audio input."""
		try:
			audio = librosa.load(
				librosa.util.buf_to_float(audio_file),
				sr = self.sample_rate
			)
			return audio
		except Exception as e:
			raise ValueError(f"Error loading audio: {str(e)}")

	def _calc_confidence_score(self, logits: torch.Tensor) -> float:
		"""Calculate confidence based on logits."""
		class_probabilities = torch.nn.functional.softmax(logits, dim=-1)
        # class_probabilities.shape = (batch_size, len(longest_audio_frame), # of token_ids in model vocab)
        # remember: each token_id in the model's vocab is the class our model is trying to predict
		confidence_score = torch.mean(torch.max(class_probabilities,dim=-1)[0]).item()
        # mean confidence score for the first sequence which holds class probabilities
        # for each entry in the class_probabilities tensor
		return confidence_score


	async def _detect_language(self, audio: np.ndarray) -> str:
		"""Detect language of input audio."""
		# TODO
		return "en"

	async def _enhance_audio(self,
		audio: np.ndarray,
		noise_reduction: bool = True,
		normalize: bool = True
	) -> np.ndarray:
		"""Apply audio enhancement techniques."""
		if normalize:
			audio = librosa.util.normalize(audio)
		if noise_reduction:
			# TODO
			pass

		return audio
