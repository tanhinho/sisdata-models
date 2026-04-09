## SISdATAModels

### FastAPI serving

This project includes a FastAPI application for serving models.

The FastAPI app lives in serving/app.py and exposes:

- GET / and GET /health for basic health checks
- POST /predict for inference (currently a placeholder implementation)

### Running the API locally

From the project root, install dependencies:

```bash
pip install -e .
```

Then start the FastAPI server with Uvicorn:

```bash
uvicorn serving.app:app --reload
```

The API will be available at <http://127.0.0.1:8000>.

Interactive API docs:

- Swagger UI: <http://127.0.0.1:8000/docs>
- ReDoc: <http://127.0.0.1:8000/redoc>
