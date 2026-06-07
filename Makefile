# AWS Bedrock RAG Chatbot Makefile

.PHONY: help install test clean deploy package lint format

# Default target
help:
	@echo "Available commands:"
	@echo "  install    - Install dependencies"
	@echo "  test       - Run tests"
	@echo "  lint       - Run linting"
	@echo "  format     - Format code"
	@echo "  clean      - Clean build artifacts"
	@echo "  package    - Package Lambda functions"
	@echo "  deploy     - Deploy infrastructure"

# Install dependencies
install:
	pip install -r bedrock_requirements.txt
	pip install -r requirements-dev.txt

# Run tests
test:
	python -m pytest tests/ -v --cov=shared --cov=lambda_functions

# Lint code
lint:
	flake8 shared/ lambda_functions/
	mypy shared/ lambda_functions/

# Format code
format:
	black shared/ lambda_functions/
	isort shared/ lambda_functions/

# Clean build artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/

# Package Lambda functions
package:
	mkdir -p build/
	cd lambda_functions/rag_orchestrator && zip -r ../../build/rag_orchestrator.zip .
	cd lambda_functions/document_indexer && zip -r ../../build/document_indexer.zip .
	cd lambda_functions/health_check && zip -r ../../build/health_check.zip .

# Deploy infrastructure (placeholder)
deploy:
	@echo "Deployment will be implemented with CloudFormation templates"
	@echo "Run: aws cloudformation deploy --template-file infrastructure/main.yaml"

# Setup development environment
setup-dev:
	cp .env.template .env
	@echo "Please edit .env file with your AWS credentials and configuration"

# Validate CloudFormation templates
validate-templates:
	aws cloudformation validate-template --template-body file://infrastructure/main.yaml

# Create S3 bucket for deployment
create-bucket:
	aws s3 mb s3://rag-traffic-vi --region us-east-1

# Sync local data to S3
sync-data:
	aws s3 sync data/ s3://rag-traffic-vi/raw/ --exclude "*.pyc" --exclude "__pycache__/*"