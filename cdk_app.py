#!/usr/bin/env python3
"""CDK app entry point for AI002 Traffic Law RAG Chatbot."""
import aws_cdk as cdk
from infrastructure.stack import ChatbotStack

app = cdk.App()

ChatbotStack(
    app,
    "AI002-ChatbotGiaothong",
    env=cdk.Environment(region="us-east-1"),
    description="AI002 - Vietnamese Traffic Law RAG Chatbot (Serverless)",
)

app.synth()
