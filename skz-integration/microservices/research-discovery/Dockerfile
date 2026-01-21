FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared code
COPY base_agent.py ./

# Copy application code
COPY app.py .

# Expose port
EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

# Run the application
CMD ["python", "app.py"]