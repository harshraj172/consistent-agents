FROM python:3.10-slim

WORKDIR /app

# Install required packages
RUN pip install --no-cache-dir pytest

# Copy task files
COPY . /app/

# Create logs directory
RUN mkdir -p /logs/verifier

# Set up execution environment
ENV PYTHONUNBUFFERED=1
