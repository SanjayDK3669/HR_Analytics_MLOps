# Use an official Python runtime as a parent image
# We'll use a specific version to ensure consistency
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the local project files into the container's /app directory
COPY . /app

# Install the dependencies from requirements.txt
COPY requirements.txt .

# Install dependencies using pip
RUN pip install --no-cache-dir -r requirements.txt

# The application is set to be run by the `app.py` script inside the `app` folder.
# We no longer need to install the 'src' directory as a package.

# Expose the port that the Flask app will run on
EXPOSE 8080

# The command to run the application using Gunicorn
# We also set the port to 8080 as required by Cloud Run
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app.app:app"]
