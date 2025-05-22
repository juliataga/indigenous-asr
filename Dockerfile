FROM python:3.10-slim

# Install deps
RUN apt-get update && apt-get install -y ffmpeg libsndfile1
