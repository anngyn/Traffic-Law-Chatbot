#!/usr/bin/env python3
"""
Setup script for AWS Bedrock RAG Chatbot
Helps initialize the project environment and dependencies
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, description):
    """Run a shell command with error handling"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 9):
        print("❌ Python 3.9 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def check_aws_cli():
    """Check if AWS CLI is installed"""
    try:
        subprocess.run(['aws', '--version'], check=True, capture_output=True)
        print("✅ AWS CLI is installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ AWS CLI not found. Please install it from: https://aws.amazon.com/cli/")
        return False

def setup_environment():
    """Setup environment file"""
    env_template = Path('.env.template')
    env_file = Path('.env')
    
    if not env_file.exists() and env_template.exists():
        shutil.copy(env_template, env_file)
        print("✅ Created .env file from template")
        print("⚠️  Please edit .env file with your AWS credentials and configuration")
        return True
    elif env_file.exists():
        print("✅ .env file already exists")
        return True
    else:
        print("❌ .env.template not found")
        return False

def install_dependencies():
    """Install Python dependencies"""
    requirements_files = [
        'bedrock_requirements.txt',
        'requirements-dev.txt'
    ]
    
    for req_file in requirements_files:
        if Path(req_file).exists():
            if not run_command(f'pip install -r {req_file}', f'Installing {req_file}'):
                return False
        else:
            print(f"⚠️  {req_file} not found, skipping")
    
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        'build',
        'logs',
        'temp'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True

def validate_project_structure():
    """Validate that all necessary files and directories exist"""
    required_files = [
        'shared/config/settings.py',
        'shared/models/data_models.py',
        'shared/utils/aws_clients.py',
        'lambda_functions/rag_orchestrator/lambda_function.py',
        'lambda_functions/document_indexer/lambda_function.py',
        'lambda_functions/health_check/lambda_function.py',
        'infrastructure/cloudformation-template.yaml',
        'bedrock_requirements.txt'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    print("✅ All required files are present")
    return True

def main():
    """Main setup function"""
    print("🚀 Setting up AWS Bedrock RAG Chatbot...")
    print("=" * 50)
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    if not check_aws_cli():
        print("⚠️  AWS CLI is recommended but not required for development")
    
    # Validate project structure
    if not validate_project_structure():
        print("❌ Project structure validation failed")
        sys.exit(1)
    
    # Setup environment
    if not setup_environment():
        print("❌ Environment setup failed")
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        print("❌ Directory creation failed")
        sys.exit(1)
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    if not install_dependencies():
        print("❌ Dependency installation failed")
        print("💡 Try running: pip install -r bedrock_requirements.txt")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Edit .env file with your AWS credentials")
    print("2. Configure your AWS profile: aws configure")
    print("3. Run tests: make test")
    print("4. Deploy infrastructure: make deploy")
    print("\n📚 For more information, see README.md")

if __name__ == "__main__":
    main()