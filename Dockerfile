# Use an official Python runtime as a parent image
# We'll use a specific version to ensure consistency
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the local project files into the container's /app directory
COPY . /app

# Install the dependencies from requirements.txt
# We use a two-step process to leverage Docker's layer caching
# First, copy only requirements.txt
COPY requirements.txt .

# Install dependencies using pip
# We need to install the specific versions of scikit-learn, etc.
RUN pip install --no-cache-dir -r requirements.txt

# The application is set to be run by the `app.py` script inside the `app` folder.
# We also need to install dependencies in the src folder
RUN pip install --no-cache-dir src/

# Expose the port that the Flask app will run on
EXPOSE 8080

# The command to run the application using Gunicorn
# This is a production-ready web server, much better than Flask's built-in server
# We also set the port to 8080 as required by Cloud Run
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app.app:app"]
