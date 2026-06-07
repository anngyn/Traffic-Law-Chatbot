"""CDK Stack: S3 + Lambda + API Gateway + EventBridge + IAM."""
from aws_cdk import (
    Duration,
    RemovalPolicy,
    Stack,
    aws_apigateway as apigw,
    aws_events as events,
    aws_events_targets as targets,
    aws_iam as iam,
    aws_lambda as _lambda,
    aws_logs as logs,
    aws_s3 as s3,
    aws_s3_notifications as s3n,
)
from constructs import Construct


class ChatbotStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs):
        super().__init__(scope, id, **kwargs)

        # ─── S3 Bucket ───────────────────────────────────────────────
        bucket = s3.Bucket(
            self, "DocsBucket",
            bucket_name=f"ai002-rag-traffic-{self.account}",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            removal_policy=RemovalPolicy.RETAIN,
            event_bridge_enabled=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
        )

        # ─── IAM: Bedrock access policy ──────────────────────────────
        bedrock_policy = iam.PolicyStatement(
            actions=[
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream",
            ],
            resources=["arn:aws:bedrock:*::foundation-model/*"],
        )

        # ─── Lambda Layer (shared code) ──────────────────────────────
        shared_layer = _lambda.LayerVersion(
            self, "SharedLayer",
            code=_lambda.Code.from_asset("shared"),
            compatible_runtimes=[_lambda.Runtime.PYTHON_3_12],
            description="Shared utilities (config, models, utils)",
        )

        # ─── Lambda: RAG Orchestrator ─────────────────────────────────
        rag_fn = _lambda.Function(
            self, "RagOrchestrator",
            function_name="ai002-rag-orchestrator",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="lambda_function.handler",
            code=_lambda.Code.from_asset("lambda_functions/rag_orchestrator"),
            memory_size=1024,
            timeout=Duration.seconds(30),
            layers=[shared_layer],
            environment={
                "S3_BUCKET_NAME": bucket.bucket_name,
                "EMBEDDING_MODEL_ID": "amazon.titan-embed-text-v2:0",
                "LLM_MODEL_ID": "anthropic.claude-3-haiku-20240307-v1:0",
                "S3_INDEX_PREFIX": "index/",
                "TOP_K_RESULTS": "5",
                "LOG_LEVEL": "INFO",
            },
            log_retention=logs.RetentionDays.ONE_MONTH,
        )
        rag_fn.add_to_role_policy(bedrock_policy)
        bucket.grant_read(rag_fn)

        # ─── Lambda: Document Indexer ─────────────────────────────────
        indexer_fn = _lambda.Function(
            self, "DocumentIndexer",
            function_name="ai002-document-indexer",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="lambda_function.handler",
            code=_lambda.Code.from_asset("lambda_functions/document_indexer"),
            memory_size=512,
            timeout=Duration.seconds(60),
            layers=[shared_layer],
            environment={
                "S3_BUCKET_NAME": bucket.bucket_name,
                "EMBEDDING_MODEL_ID": "amazon.titan-embed-text-v2:0",
                "S3_INDEX_PREFIX": "index/",
                "S3_RAW_PREFIX": "raw/",
                "LOG_LEVEL": "INFO",
            },
            log_retention=logs.RetentionDays.ONE_MONTH,
        )
        indexer_fn.add_to_role_policy(bedrock_policy)
        bucket.grant_read_write(indexer_fn)

        # ─── Lambda: Health Check ─────────────────────────────────────
        health_fn = _lambda.Function(
            self, "HealthCheck",
            function_name="ai002-health-check",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="lambda_function.handler",
            code=_lambda.Code.from_asset("lambda_functions/health_check"),
            memory_size=128,
            timeout=Duration.seconds(10),
            environment={
                "S3_BUCKET_NAME": bucket.bucket_name,
            },
            log_retention=logs.RetentionDays.ONE_MONTH,
        )
        bucket.grant_read(health_fn)

        # ─── EventBridge: S3 upload → trigger indexer ─────────────────
        events.Rule(
            self, "S3UploadRule",
            rule_name="ai002-s3-doc-upload",
            event_pattern=events.EventPattern(
                source=["aws.s3"],
                detail_type=["Object Created"],
                detail={
                    "bucket": {"name": [bucket.bucket_name]},
                    "object": {"key": [{"prefix": "raw/"}]},
                },
            ),
            targets=[targets.LambdaFunction(indexer_fn)],
        )

        # ─── API Gateway ──────────────────────────────────────────────
        api = apigw.RestApi(
            self, "ChatbotApi",
            rest_api_name="ai002-traffic-rag",
            description="AI002 Vietnamese Traffic Law RAG API",
            deploy_options=apigw.StageOptions(stage_name="prod"),
            default_cors_preflight_options=apigw.CorsOptions(
                allow_origins=apigw.Cors.ALL_ORIGINS,
                allow_methods=apigw.Cors.ALL_METHODS,
            ),
        )

        # POST /chat → rag_orchestrator
        chat_resource = api.root.add_resource("chat")
        chat_resource.add_method(
            "POST",
            apigw.LambdaIntegration(rag_fn),
        )

        # GET /health → health_check
        health_resource = api.root.add_resource("health")
        health_resource.add_method(
            "GET",
            apigw.LambdaIntegration(health_fn),
        )
