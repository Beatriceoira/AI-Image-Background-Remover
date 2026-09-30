FROM python:3.11-slim

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file
COPY requirements.txt requirements-gpu.txt .

# Install the dependencies
RUN pip install -r requirements-gpu.txt

# Copy the application code
COPY . .

# Expose the port
EXPOSE 8000

# Run the command to start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]