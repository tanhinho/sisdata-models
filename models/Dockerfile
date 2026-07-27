FROM python:3.13-slim

# Install uv
RUN pip install uv

WORKDIR /app
COPY models/* .
COPY ../pyproject.toml .

# Sync dependencies
RUN uv sync --frozen

# Run the pipeline. 
CMD uv run python run_models.py