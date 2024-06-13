# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Update and install awscli
RUN apt update -y && apt install awscli -y

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Run app.py when the container launches
CMD ["python3", "app.py"]

# Build the Docker Image:
#docker build -t my-flask-app .


# Run the Docker Container:
# docker run -p 8080:8080 my-flask-app
 