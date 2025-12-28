# Minimal Docker image for RAG Testcase Generator API
FROM python:3.9-slim

WORKDIR /app

# Copy only essential files
COPY requirements.txt .
COPY api/ ./api/
COPY src/ ./src/
COPY single_hop.env .
COPY multi_hop.env .
COPY .env .

# Install dependencies without cache
RUN pip install --no-cache-dir -r requirements.txt \
    && mkdir -p runs src/outputs src/outputs_multihop

# Expose API port
EXPOSE 10500

# Run API
CMD ["python", "-m", "api.main"]
