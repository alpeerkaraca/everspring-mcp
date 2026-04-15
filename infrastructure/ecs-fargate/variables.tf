variable "aws_region" {
  description = "AWS region for all resources."
  type        = string
  default     = "eu-central-1"
}

variable "name_prefix" {
  description = "Prefix used for naming AWS resources."
  type        = string
  default     = "everspring"
}

variable "s3_bucket_name" {
  description = "Prefix for S3 bucket names (must be globally unique)."
  type        = string
  default     = "everspring-mcp-kb"
  
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC."
  type        = string
  default     = "10.30.0.0/16"
}

variable "public_subnet_cidrs" {
  description = "Two CIDR blocks for public subnets (one per AZ)."
  type        = list(string)
  default     = ["10.30.0.0/24", "10.30.1.0/24"]
}

variable "private_subnet_cidrs" {
  description = "Two CIDR blocks for private subnets (one per AZ)."
  type        = list(string)
  default     = ["10.30.10.0/24", "10.30.11.0/24"]
}

variable "domain_name" {
  description = "Fully qualified domain for HTTPS endpoint (DNS managed externally, e.g. Cloudflare)."
  type        = string
  default     = "everspring.example.com"
}

variable "container_image" {
  description = "Container image URI to run on ECS (if null, defaults to ECR repo URL with :latest)."
  type        = string
  default     = null
}

variable "container_name" {
  description = "Name of the application container."
  type        = string
  default     = "everspring-mcp"
}

variable "container_port" {
  description = "Application HTTP port exposed by the container."
  type        = number
  default     = 8080
}

variable "task_cpu" {
  description = "Fargate task CPU units (1024 = 1 vCPU)."
  type        = number
  default     = 2048
}

variable "task_memory" {
  description = "Fargate task memory in MiB (4096 = 4GB)."
  type        = number
  default     = 6144
}

variable "desired_count" {
  description = "Desired number of ECS tasks."
  type        = number
  default     = 1
}

variable "task_command" {
  description = "Container command override used by ECS."
  type        = list(string)
  default = [
    "python",
    "-m",
    "everspring_mcp.main",
    "serve",
    "--transport",
    "http",
    "--tier",
    "slim",
  ]
}

variable "snapshot_bucket_name" {
  description = "S3 bucket that stores snapshots consumed by the app."
  type        = string
  default     = null
}

variable "snapshot_prefix" {
  description = "Optional S3 prefix for snapshot objects."
  type        = string
  default     = "spring-docs"
}

variable "acm_certificate_transparency_logging_preference" {
  description = "ACM certificate transparency setting."
  type        = string
  default     = "ENABLED"
}

variable "health_check_path" {
  description = "ALB target group health check path."
  type        = string
  default     = "/healthz"
}

variable "tags" {
  description = "Common tags applied to all resources."
  type        = map(string)
  default     = {}
}
