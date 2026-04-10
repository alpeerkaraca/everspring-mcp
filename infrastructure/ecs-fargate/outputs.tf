output "vpc_id" {
  description = "VPC ID hosting ALB, ECS and EFS."
  value       = aws_vpc.this.id
}

output "public_subnet_ids" {
  description = "Public subnet IDs used by the ALB."
  value       = values(aws_subnet.public)[*].id
}

output "private_subnet_ids" {
  description = "Private subnet IDs used by ECS tasks and EFS mount targets."
  value       = values(aws_subnet.private)[*].id
}

output "alb_dns_name" {
  description = "ALB DNS name."
  value       = aws_lb.this.dns_name
}

output "app_url" {
  description = "Primary HTTPS URL for the service."
  value       = "https://${var.domain_name}"
}

output "cloudflare_dns_target" {
  description = "ALB DNS name to use as CNAME/Proxy target in Cloudflare for the app domain."
  value       = aws_lb.this.dns_name
}

output "ecr_repository_url" {
  description = "ECR repository URL for pushing the container image."
  value       = aws_ecr_repository.app.repository_url
}

output "ecs_cluster_name" {
  description = "ECS cluster name."
  value       = aws_ecs_cluster.this.name
}

output "ecs_service_name" {
  description = "ECS service name."
  value       = aws_ecs_service.this.name
}

output "ecs_task_definition_arn" {
  description = "Current task definition ARN."
  value       = aws_ecs_task_definition.this.arn
}

output "efs_file_system_id" {
  description = "EFS file system ID used for persistent cache."
  value       = aws_efs_file_system.cache.id
}

output "efs_access_point_id" {
  description = "EFS access point mounted by ECS task."
  value       = aws_efs_access_point.cache_path.id
}

output "acm_certificate_arn" {
  description = "ACM certificate ARN configured on the HTTPS listener."
  value       = aws_acm_certificate.this.arn
}

output "acm_dns_validation_records" {
  description = "Create these DNS records in Cloudflare to validate ACM certificate ownership."
  value = [
    for dvo in aws_acm_certificate.this.domain_validation_options : {
      domain = dvo.domain_name
      name   = dvo.resource_record_name
      type   = dvo.resource_record_type
      value  = dvo.resource_record_value
    }
  ]
}
