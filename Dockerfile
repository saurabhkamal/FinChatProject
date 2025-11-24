# Use BuildKit syntax for caching support
# syntax=docker/dockerfile:1.4

FROM python:3.10-slim-buster AS base

WORKDIR /app

# Step 1: Copy only requirements to leverage Docker layer caching
COPY requirements.txt /app/

# Install dependencies now
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Step 2: Copy application code
COPY . /app

# (Optional) If you have local module install from . (editable), you may run:
# RUN pip install -e .

# Set environment variables (if any) for production
ENV PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1

# Use a lighter entrypoint
CMD ["python3", "app.py"]
