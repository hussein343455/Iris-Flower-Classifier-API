# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file first and install them
# Now copy the rest of your application code
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

# Command to run the application
# Use 0.0.0.0 to make it accessible from outside the container
CMD ["uvicorn", "Fastapi_test:app", "--host", "0.0.0.0", "--port", "80"]