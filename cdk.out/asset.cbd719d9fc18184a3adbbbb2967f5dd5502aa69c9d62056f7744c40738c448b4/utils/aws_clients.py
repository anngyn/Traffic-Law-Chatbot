"""
AWS client utilities for Bedrock RAG Chatbot
Provides centralized AWS service client management
"""

import boto3
import logging
from typing import Optional, Dict, Any
from botocore.exceptions import ClientError, NoCredentialsError
from botocore.config import Config

from ..config.settings import get_config

logger = logging.getLogger(__name__)

class AWSClientManager:
    """Manages AWS service clients with proper configuration and error handling"""
    
    def __init__(self):
        self.config = get_config()
        self._clients: Dict[str, Any] = {}
        
        # Configure boto3 with retry settings
        self._boto_config = Config(
            region_name=self.config.aws.region,
            retries={
                'max_attempts': 3,
                'mode': 'adaptive'
            },
            max_pool_connections=50
        )
    
    def get_bedrock_client(self) -> Optional[boto3.client]:
        """Get Bedrock Runtime client"""
        if 'bedrock' not in self._clients:
            try:
                self._clients['bedrock'] = boto3.client(
                    'bedrock-runtime',
                    region_name=self.config.bedrock.region,
                    config=self._boto_config
                )
                logger.info("Bedrock client initialized successfully")
            except (ClientError, NoCredentialsError) as e:
                logger.error(f"Failed to initialize Bedrock client: {e}")
                return None
        
        return self._clients['bedrock']
    
    def get_s3_client(self) -> Optional[boto3.client]:
        """Get S3 client"""
        if 's3' not in self._clients:
            try:
                self._clients['s3'] = boto3.client(
                    's3',
                    region_name=self.config.aws.region,
                    config=self._boto_config
                )
                logger.info("S3 client initialized successfully")
            except (ClientError, NoCredentialsError) as e:
                logger.error(f"Failed to initialize S3 client: {e}")
                return None
        
        return self._clients['s3']
    
    def get_eventbridge_client(self) -> Optional[boto3.client]:
        """Get EventBridge client"""
        if 'eventbridge' not in self._clients:
            try:
                self._clients['eventbridge'] = boto3.client(
                    'events',
                    region_name=self.config.aws.region,
                    config=self._boto_config
                )
                logger.info("EventBridge client initialized successfully")
            except (ClientError, NoCredentialsError) as e:
                logger.error(f"Failed to initialize EventBridge client: {e}")
                return None
        
        return self._clients['eventbridge']
    
    def get_cloudwatch_client(self) -> Optional[boto3.client]:
        """Get CloudWatch client"""
        if 'cloudwatch' not in self._clients:
            try:
                self._clients['cloudwatch'] = boto3.client(
                    'cloudwatch',
                    region_name=self.config.aws.region,
                    config=self._boto_config
                )
                logger.info("CloudWatch client initialized successfully")
            except (ClientError, NoCredentialsError) as e:
                logger.error(f"Failed to initialize CloudWatch client: {e}")
                return None
        
        return self._clients['cloudwatch']
    
    def health_check(self) -> Dict[str, bool]:
        """Perform health checks on AWS services"""
        health_status = {}
        
        # Check S3
        s3_client = self.get_s3_client()
        if s3_client:
            try:
                s3_client.head_bucket(Bucket=self.config.s3.bucket_name)
                health_status['s3'] = True
                logger.info("S3 health check passed")
            except ClientError as e:
                health_status['s3'] = False
                logger.error(f"S3 health check failed: {e}")
        else:
            health_status['s3'] = False
        
        # Check Bedrock
        bedrock_client = self.get_bedrock_client()
        if bedrock_client:
            try:
                # Simple test to check if Bedrock is accessible
                bedrock_client.list_foundation_models()
                health_status['bedrock'] = True
                logger.info("Bedrock health check passed")
            except ClientError as e:
                health_status['bedrock'] = False
                logger.error(f"Bedrock health check failed: {e}")
        else:
            health_status['bedrock'] = False
        
        return health_status

# Global client manager instance
_client_manager = None

def get_aws_clients() -> AWSClientManager:
    """Get the global AWS client manager instance"""
    global _client_manager
    if _client_manager is None:
        _client_manager = AWSClientManager()
    return _client_manager