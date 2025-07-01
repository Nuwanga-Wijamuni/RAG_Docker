# FINAL, OPTIMIZED DOCKERFILE

# Use the small, official Python slim image
FROM python:3.12.4-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies needed by PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy your requirements file
COPY requirements.txt .

# --- THE OPTIMIZED INSTALLATION ---
# 1. First, install the CPU-only version of PyTorch. This is much smaller and faster.
#    This satisfies the 'torch' dependency before the next step.
RUN pip install --no-cache-dir --timeout 1000 torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu

# 2. Now, install the rest of the requirements.
#    pip will see that 'torch' is already installed and will skip it, saving massive amounts of data.
RUN pip install --no-cache-dir --timeout 1000 -r requirements.txt

# Copy your application code into the container
COPY RAGpipline.py .

# Expose the port and set the run command
EXPOSE 8081
ENV PORT=8081
CMD ["uvicorn", "RAGpipline:app", "--host", "0.0.0.0", "--port", "8081"]