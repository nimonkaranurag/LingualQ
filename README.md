# LingualQ: (Multi-Modal) Enterprise Translation Memory Management System with AI-Driven QA

A robust task automation system leveraging cutting-edge AI for managing translation memory, transcription, and email automation. This system is built on a microservices architecture, offering real-time processing for text and audio inputs.

---

## Key Features

- **Multi-Modal Input Processing:** Handles both text and audio inputs efficiently.
- **Translation Memory Management:** Provides semantic search and memory lookup.
- **Real-Time Quality Assessment:** Ensures output meets professional standards.
- **Asynchronous Task Orchestration:** Manages tasks dynamically with real-time status updates.
- **Automated Email Generation:** Converts meeting transcripts into well-structured emails.
- **Cross-Lingual Semantic Similarity:** Delivers accurate translations across languages.

---

## Technical Architecture

### Core Components

#### 1. Task Orchestration System
Implemented in `services/task_orchestrator.py`, this module coordinates asynchronous task processing:
- Tasks include transcription, translation, summarization, and email generation.
- State management for each task:

```python
class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
```

#### 2. Language Model Interface
`services/llm_interface.py` interfaces with multiple AI models:
- **BART:** Summarization
- **GPT-2:** Text generation
- **MarianMT:** Translation

Capabilities include:
- Meeting summarization
- Action item extraction
- Email template generation
- Cross-lingual translation

#### 3. Translation Memory Management
`services/translations_memory_manager.py` powers efficient translation retrieval:
- Embedding computation using BERT
- Cosine similarity-based lookups
- Quality scoring and fuzzy matching

```python
def compute_embeddings(self, text: str) -> np.ndarray:
    inputs = self.tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = self.model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()
```

#### 4. Audio Processing Pipeline
`services/audio_processor.py` facilitates:
- Audio transcription using Wav2Vec2
- Language detection and confidence scoring
- Pre-processing for enhanced transcription quality

---

### API Layer

Exposed RESTful endpoints using FastAPI:

1. **Audio Endpoints (`/api/v1/audio/`):**
   - Transcription initiation
   - Task status tracking
   - Quality assessment

2. **Translation Endpoints (`/api/v1/translation/`):**
   - Text translation
   - Memory lookup
   - Quality evaluation

3. **Email Endpoints (`/api/v1/email/`):**
   - Meeting summaries
   - Action item extractions
   - Email generation

---

## Deployment

### Dockerized Setup
- Micro-services architecture for scalable deployment.
- Redis for managing task queues.
- Nginx for serving the frontend.
- Integrated health checks for container monitoring.

### Steps to Deploy
-------------------------------------------------------------------

1. Clone the repository:
   -- this is an ongoing project, contributions are welcome --
---

## Project Structure

```
LingualQ/
├── docker/
│   ├── docker-compose.yml
│   ├── Dockerfile
│   └── nginx.conf
├── src/
│   ├── api/
│   │   ├── endpoints/
│   │   ├── models/
│   │   └── main.py
│   ├── core/
│   │   ├── config.py
│   │   └── logging.py
│   └── services/
│       ├── audio_processor.py
│       ├── llm_interface.py
│       ├── quality_checker.py
│       ├── task_orchestrator.py
│       └── translations_memory_manager.py
└── tests/
```

---

## API Documentation

Interactive documentation available at `http://localhost:8000/docs` when the app is running.

---

## Future Enhancements

1. **Scalability:**
   - Horizontal scaling and load balancing
   - Enhanced caching mechanisms

2. **Security:**
   - OAuth2 authentication
   - Rate limiting

3. **Monitoring:**
   - Prometheus for metrics
   - ELK stack for logging

---

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

---

## License

[MIT License](LICENSE)
