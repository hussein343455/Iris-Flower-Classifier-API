# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# 1. Copy only the requirements file
COPY requirements.txt .

# 2. Install dependencies (this layer will be cached)
RUN pip install --no-cache-dir -r requirements.txt

# 3. Now copy the rest of the application code
COPY . .

# Command to run the application
CMD ["uvicorn", "Fastapi_test:app", "--host", "0.0.0.0", "--port", "80"]