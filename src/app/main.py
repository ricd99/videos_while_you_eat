"""
FASTAPI + (GRADIO SERVING APPLICATION - soon!)

This application provides a complete serving solution for the ytrec model
with both programmatic API access and (a user-friendly web interface - soon!).

Architecture:
- FastAPI: High-performance REST API with automatic OpenAPI documentation
- (Gradio: User-friendly web UI for manual testing and demonstrations - soon!)
- Pydantic: Data validation and automatic API documentation
"""

from fastapi import FastAPI
from pydantic import BaseModel
from src.serving.inference import predict

app = FastAPI(
    title = "ytrec prediction API"
    description = "ML API for showing related long-form yt channels based on a query channel"
    version = "1.0.0"
)


@app.get("/")
def root():
    """
    Health check endpoint for AWS Application Load Balancer
    """
    return {"status": "ok"}

class ChannelData(BaseModel):
    """
    Yt channel data schema
    """

    channel_id: str
    channel_name: str
    title: str
    description: str
    country: str
    topics: list
    keywords: str
    uploads: str
    videos: list