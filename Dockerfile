# Production image. Python 3.14 slim, non-root, gunicorn serving the app factory.
# The GROQ_API_KEY is provided at runtime by the platform (e.g. Render env var).
# .env is never copied in (see .dockerignore).
FROM python:3.14-slim

# Install dependencies first for better layer caching. gunicorn is installed
# here (production only) and deliberately kept out of requirements.txt.
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy only what the service needs to run.
COPY app/ ./app/
COPY prompts/ ./prompts/
COPY run.py .

# Run as a non-root user.
RUN useradd --create-home --uid 1000 appuser
USER appuser

# Documented port; Render (or any host) may override via $PORT.
EXPOSE 8000
ENV PORT=8000

# Serve the Flask app factory. --timeout 120 accommodates slower LLM calls.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8000} --workers 2 --timeout 120 'app:create_app()'"]
