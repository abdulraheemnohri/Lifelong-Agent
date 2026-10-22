# Sandboxed execution image for the Lifelong Self-Evolving Offline Agent.
# Generated tools run inside this container in production deployments;
# AST validation alone is NOT a full sandbox (see SAFETY.md).

FROM python:3.11-slim

# git is required for the agent's self-modification history and rollback.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-optional.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

# Non-root user for defence in depth.
RUN useradd --create-home --shell /bin/bash agent
RUN mkdir -p /home/agent/workspace && chown -R agent:agent /home/agent
USER agent
WORKDIR /home/agent/workspace

# Model artifacts and state live in the mounted workspace volume.
ENV WORKSPACE=/home/agent/workspace
VOLUME ["/home/agent/workspace"]

# Default: run the offline agent headless with the safe mock fallback.
CMD ["python", "/app/scripts/run_headless.py", "--task", "Introduce yourself and list your tools"]
