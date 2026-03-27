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