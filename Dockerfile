# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install uv (dependency manager used by this repo)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies into a project-local .venv via uv (no bare pip/venv)
RUN uv sync --frozen --no-dev

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Serve the FastAPI app via uvicorn. --frozen --no-dev keeps this from
# re-resolving/downloading the dev dependency group (dvc, ruff, pytest, ...)
# at container start — the env was already synced at build time above.
CMD ["uv", "run", "--frozen", "--no-dev", "uvicorn", "serving.api.main:app", "--host", "0.0.0.0", "--port", "8080"]

# Build the Docker Image:
#docker build -t production-ml-platform .


# Run the Docker Container:
# docker run -p 8080:8080 production-ml-platform
 