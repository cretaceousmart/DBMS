# Use the official Python base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the project into the container
COPY . /app

# Install necessary build dependencies and clean up
RUN apt-get update && apt-get install -y gcc python3-dev && apt-get clean

# Copy and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Install Jupyter Notebook
RUN pip install jupyter

# Set the default command to run Jupyter Notebook
CMD ["jupyter", "notebook", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]
