FROM python:3.12-slim

# Create a non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Set working directory
WORKDIR /app

# Copy requirements.txt first for better caching
COPY requirements.txt .

# Install system dependencies for PDF processing
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Change ownership of the app directory to the non-root user
RUN chown -R appuser:appgroup /app

# Switch to the non-root user
USER appuser

# Make the main.py file executable
RUN chmod +x src/main.py

# Set the entrypoint to python and default command to show usage
ENTRYPOINT ["python", "main.py"]