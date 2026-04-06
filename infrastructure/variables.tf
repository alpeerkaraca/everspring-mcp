variable "aws_region" {
  description = "AWS region for all resources."
  type        = string
  default     = "us-east-1"
}

variable "registry_bucket_name" {
  description = "Name of the S3 bucket that stores scraper registry artifacts."
  type        = string
}

variable "registry_bucket_force_destroy" {
  description = "Allow Terraform to delete a non-empty registry bucket."
  type        = bool
  default     = false
}

variable "registry_prefix" {
  description = "S3 prefix for documentation and manifest artifacts."
  type        = string
  default     = "docs"

  validation {
    condition     = length(trim(var.registry_prefix, "/")) > 0
    error_message = "registry_prefix must not be empty or only slashes."
  }
}

variable "lambda_function_name" {
  description = "AWS Lambda function name for scraper execution."
  type        = string
  default     = "everspring-scraper"
}

variable "lambda_runtime" {
  description = "Lambda runtime for scraper execution package."
  type        = string
  default     = "python3.11"
}

variable "lambda_handler" {
  description = "Lambda handler entry point in the deployment package."
  type        = string
  default     = "lambda_function.lambda_handler"
}

variable "lambda_architectures" {
  description = "Lambda CPU architecture list."
  type        = list(string)
  default     = ["x86_64"]
}

variable "lambda_artifact_bucket" {
  description = "Optional S3 bucket containing the Lambda deployment package. Defaults to registry bucket."
  type        = string
  default     = null
}

variable "lambda_artifact_key" {
  description = "S3 key for the Lambda deployment ZIP artifact."
  type        = string
}

variable "lambda_artifact_object_version" {
  description = "Optional S3 object version of the Lambda deployment package."
  type        = string
  default     = null
}

variable "lambda_source_code_hash" {
  description = "Optional base64-encoded SHA256 hash of deployment package for deterministic updates."
  type        = string
  default     = null
}

variable "lambda_timeout_seconds" {
  description = "Lambda timeout in seconds."
  type        = number
  default     = 900

  validation {
    condition     = var.lambda_timeout_seconds >= 1 && var.lambda_timeout_seconds <= 900
    error_message = "lambda_timeout_seconds must be between 1 and 900."
  }
}

variable "lambda_memory_size" {
  description = "Lambda memory size in MB."
  type        = number
  default     = 2048

  validation {
    condition     = var.lambda_memory_size >= 128 && var.lambda_memory_size <= 10240
    error_message = "lambda_memory_size must be between 128 and 10240."
  }
}

variable "lambda_ephemeral_storage_mb" {
  description = "Lambda ephemeral storage size in MB. Playwright requires higher temporary storage."
  type        = number
  default     = 2048

  validation {
    condition     = var.lambda_ephemeral_storage_mb >= 512 && var.lambda_ephemeral_storage_mb <= 10240
    error_message = "lambda_ephemeral_storage_mb must be between 512 and 10240."
  }
}

variable "lambda_reserved_concurrency" {
  description = "Optional reserved concurrency for Lambda (null keeps unreserved)."
  type        = number
  default     = null
}

variable "lambda_environment" {
  description = "Additional environment variables for the scraper Lambda."
  type        = map(string)
  default     = {}
}

variable "playwright_layer_name" {
  description = "Name of the Lambda layer that provides Playwright-compatible binaries."
  type        = string
  default     = "everspring-playwright-layer"
}

variable "playwright_layer_artifact_bucket" {
  description = "S3 bucket containing the Playwright Lambda layer ZIP artifact (for example built with playwright-aws-lambda)."
  type        = string
}

variable "playwright_layer_artifact_key" {
  description = "S3 key for the Playwright Lambda layer ZIP artifact."
  type        = string
}

variable "playwright_layer_artifact_object_version" {
  description = "Optional S3 object version for the Playwright Lambda layer artifact."
  type        = string
  default     = null
}

variable "playwright_layer_compatible_runtimes" {
  description = "Compatible runtimes published on the Playwright layer."
  type        = list(string)
  default     = ["python3.11"]
}

variable "log_retention_days" {
  description = "CloudWatch log retention period for scraper Lambda logs."
  type        = number
  default     = 30
}

variable "tags" {
  description = "Common tags applied to all infrastructure resources."
  type        = map(string)
  default     = {}
}

