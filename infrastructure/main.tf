terraform {
  required_version = ">= 1.6.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

data "aws_partition" "current" {}

locals {
  registry_prefix        = trim(var.registry_prefix, "/")
  lambda_artifact_bucket = coalesce(var.lambda_artifact_bucket, var.registry_bucket_name)
  registry_bucket_arn    = "arn:${data.aws_partition.current.partition}:s3:::${aws_s3_bucket.registry.id}"
  registry_objects_arn   = "arn:${data.aws_partition.current.partition}:s3:::${aws_s3_bucket.registry.id}/${local.registry_prefix}/*"
}

resource "aws_s3_bucket" "registry" {
  bucket        = var.registry_bucket_name
  force_destroy = var.registry_bucket_force_destroy

  tags = var.tags
}

resource "aws_s3_bucket_public_access_block" "registry" {
  bucket = aws_s3_bucket.registry.id

  block_public_acls       = true
  ignore_public_acls      = true
  block_public_policy     = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "registry" {
  bucket = aws_s3_bucket.registry.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "registry" {
  bucket = aws_s3_bucket.registry.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_ownership_controls" "registry" {
  bucket = aws_s3_bucket.registry.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

data "aws_iam_policy_document" "registry_tls_only" {
  statement {
    sid    = "DenyInsecureTransport"
    effect = "Deny"

    principals {
      type        = "*"
      identifiers = ["*"]
    }

    actions = ["s3:*"]
    resources = [
      local.registry_bucket_arn,
      "${local.registry_bucket_arn}/*",
    ]

    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}

resource "aws_s3_bucket_policy" "registry" {
  bucket = aws_s3_bucket.registry.id
  policy = data.aws_iam_policy_document.registry_tls_only.json
}

data "aws_iam_policy_document" "lambda_assume_role" {
  statement {
    sid     = "LambdaAssumeRole"
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "scraper_lambda" {
  name               = "${var.lambda_function_name}-role"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume_role.json
  tags               = var.tags
}

resource "aws_cloudwatch_log_group" "scraper_lambda" {
  name              = "/aws/lambda/${var.lambda_function_name}"
  retention_in_days = var.log_retention_days
  tags              = var.tags
}

data "aws_iam_policy_document" "scraper_lambda_least_privilege" {
  statement {
    sid    = "WriteAndReadRegistryObjects"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
    ]
    resources = [local.registry_objects_arn]
  }

  statement {
    sid    = "ListRegistryPrefix"
    effect = "Allow"
    actions = [
      "s3:ListBucket",
    ]
    resources = [local.registry_bucket_arn]

    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values = [
        local.registry_prefix,
        "${local.registry_prefix}/*",
      ]
    }
  }

  statement {
    sid    = "WriteCloudWatchLogs"
    effect = "Allow"
    actions = [
      "logs:CreateLogStream",
      "logs:PutLogEvents",
    ]
    resources = ["${aws_cloudwatch_log_group.scraper_lambda.arn}:*"]
  }
}

resource "aws_iam_role_policy" "scraper_lambda_least_privilege" {
  name   = "${var.lambda_function_name}-least-privilege"
  role   = aws_iam_role.scraper_lambda.id
  policy = data.aws_iam_policy_document.scraper_lambda_least_privilege.json
}

resource "aws_lambda_layer_version" "playwright" {
  layer_name  = var.playwright_layer_name
  description = "Playwright Chromium runtime layer for AWS Lambda. Provide an artifact built for Lambda (for example using playwright-aws-lambda)."

  s3_bucket = var.playwright_layer_artifact_bucket
  s3_key    = var.playwright_layer_artifact_key

  s3_object_version        = var.playwright_layer_artifact_object_version
  compatible_runtimes      = var.playwright_layer_compatible_runtimes
  compatible_architectures = var.lambda_architectures
}

resource "aws_lambda_function" "scraper" {
  function_name = var.lambda_function_name
  role          = aws_iam_role.scraper_lambda.arn
  runtime       = var.lambda_runtime
  handler       = var.lambda_handler

  s3_bucket         = local.lambda_artifact_bucket
  s3_key            = var.lambda_artifact_key
  s3_object_version = var.lambda_artifact_object_version
  source_code_hash  = var.lambda_source_code_hash

  memory_size   = var.lambda_memory_size
  timeout       = var.lambda_timeout_seconds
  architectures = var.lambda_architectures

  reserved_concurrent_executions = var.lambda_reserved_concurrency

  ephemeral_storage {
    size = var.lambda_ephemeral_storage_mb
  }

  layers = [aws_lambda_layer_version.playwright.arn]

  environment {
    variables = merge(
      {
        EVERSPRING_S3_BUCKET = aws_s3_bucket.registry.id
        EVERSPRING_S3_PREFIX = local.registry_prefix
        AWS_REGION           = var.aws_region
        PYTHONUNBUFFERED     = "1"
      },
      var.lambda_environment
    )
  }

  depends_on = [aws_cloudwatch_log_group.scraper_lambda]
  tags       = var.tags
}

