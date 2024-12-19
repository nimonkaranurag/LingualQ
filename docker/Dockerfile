FROM python:3.9-slim

WORKDIR app/

RUN apt-get update && apt-get install -y \
	build-essential \
	libffi-dev \
	&& rm -rf /var/lib/apt/lists/*

RUN python3 -m venv .venv/
ENV PATH="/app/.venv/bin:$PATH"

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
COPY tests/ tests/

EXPOSE 8000

CMD["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", 8000] 
