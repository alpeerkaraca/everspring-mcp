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

data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

locals {
  azs = slice(data.aws_availability_zones.available.names, 0, 2)

  common_tags = merge(
    {
      Project     = "everspring-mcp"
      ManagedBy   = "terraform"
      Environment = "prod"
    },
    var.tags
  )

  ecr_repository_name = "${var.name_prefix}-mcp"
  ecs_cluster_name    = "${var.name_prefix}-ecs-cluster"
  ecs_service_name    = "${var.name_prefix}-mcp-service"
  task_family_name    = "${var.name_prefix}-mcp-task"
  alb_name            = "${var.name_prefix}-alb"
  efs_name            = "${var.name_prefix}-efs-cache"

  app_image = coalesce(var.container_image, "${aws_ecr_repository.app.repository_url}:latest")

  snapshot_bucket_arn = var.snapshot_bucket_name == null ? null : "arn:${data.aws_partition.current.partition}:s3:::${var.snapshot_bucket_name}"
  snapshot_objects_arn = var.snapshot_bucket_name == null ? null : (
    trim(var.snapshot_prefix, "/") == ""
    ? "${local.snapshot_bucket_arn}/*"
    : "${local.snapshot_bucket_arn}/${trim(var.snapshot_prefix, "/")}/*"
  )
}

resource "aws_vpc" "this" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = merge(local.common_tags, { Name = "${var.name_prefix}-vpc" })
}

resource "aws_internet_gateway" "this" {
  vpc_id = aws_vpc.this.id

  tags = merge(local.common_tags, { Name = "${var.name_prefix}-igw" })
}

resource "aws_subnet" "public" {
  for_each = {
    for idx, cidr in var.public_subnet_cidrs : idx => cidr
  }

  vpc_id                  = aws_vpc.this.id
  cidr_block              = each.value
  availability_zone       = local.azs[tonumber(each.key)]
  map_public_ip_on_launch = true

  tags = merge(local.common_tags, { Name = "${var.name_prefix}-public-${each.key}" })
}

resource "aws_subnet" "private" {
  for_each = {
    for idx, cidr in var.private_subnet_cidrs : idx => cidr
  }

  vpc_id                  = aws_vpc.this.id
  cidr_block              = each.value
  availability_zone       = local.azs[tonumber(each.key)]
  map_public_ip_on_launch = false

  tags = merge(local.common_tags, { Name = "${var.name_prefix}-private-${each.key}" })
}

resource "aws_eip" "nat" {
  domain = "vpc"

  tags = merge(local.common_tags, { Name = "${var.name_prefix}-nat-eip" })
}

resource "aws_nat_gateway" "this" {
  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public["0"].id

  depends_on = [aws_internet_gateway.this]

  tags = merge(local.common_tags, { Name = "${var.name_prefix}-nat" })
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.this.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.this.id
  }

  tags = merge(local.common_tags, { Name = "${var.name_prefix}-public-rt" })
}

resource "aws_route_table_association" "public" {
  for_each = aws_subnet.public

  subnet_id      = each.value.id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.this.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.this.id
  }

  tags = merge(local.common_tags, { Name = "${var.name_prefix}-private-rt" })
}

resource "aws_route_table_association" "private" {
  for_each = aws_subnet.private

  subnet_id      = each.value.id
  route_table_id = aws_route_table.private.id
}

# ALB security group: internet ingress on 80/443 only.
resource "aws_security_group" "alb" {
  name        = "${var.name_prefix}-alb-sg"
  description = "Allow HTTP/HTTPS from internet to ALB"
  vpc_id      = aws_vpc.this.id

  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    description = "Allow outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, { Name = "${var.name_prefix}-alb-sg" })
}

# ECS security group: app port is reachable only from ALB.
resource "aws_security_group" "ecs" {
  name        = "${var.name_prefix}-ecs-sg"
  description = "Allow app traffic from ALB only"
  vpc_id      = aws_vpc.this.id

  ingress {
    description     = "App traffic from ALB"
    from_port       = var.container_port
    to_port         = var.container_port
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    description = "Allow outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, { Name = "${var.name_prefix}-ecs-sg" })
}

# EFS security group: NFS access only from ECS tasks.
resource "aws_security_group" "efs" {
  name        = "${var.name_prefix}-efs-sg"
  description = "Allow NFS from ECS tasks only"
  vpc_id      = aws_vpc.this.id

  ingress {
    description     = "NFS from ECS"
    from_port       = 2049
    to_port         = 2049
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs.id]
  }

  egress {
    description = "Allow outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, { Name = "${var.name_prefix}-efs-sg" })
}

resource "aws_efs_file_system" "cache" {
  encrypted = true

  tags = merge(local.common_tags, { Name = local.efs_name })
}

resource "aws_efs_mount_target" "private" {
  for_each = aws_subnet.private

  file_system_id  = aws_efs_file_system.cache.id
  subnet_id       = each.value.id
  security_groups = [aws_security_group.efs.id]
}

# Access point enforces expected UID/GID and path ownership for non-root container user.
resource "aws_efs_access_point" "cache_path" {
  file_system_id = aws_efs_file_system.cache.id

  posix_user {
    gid = 1000
    uid = 1000
  }

  root_directory {
    path = "/everspring-cache"
    creation_info {
      owner_gid   = 1000
      owner_uid   = 1000
      permissions = "0750"
    }
  }

  tags = merge(local.common_tags, { Name = "${var.name_prefix}-efs-ap" })
}

resource "aws_ecr_repository" "app" {
  name                 = local.ecr_repository_name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  tags = local.common_tags
}

resource "aws_ecs_cluster" "this" {
  name = local.ecs_cluster_name

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = local.common_tags
}

resource "aws_cloudwatch_log_group" "ecs" {
  name              = "/ecs/${local.task_family_name}"
  retention_in_days = 30

  tags = local.common_tags
}



data "aws_iam_policy_document" "ecs_task_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "ecs_s3_access" {
  statement {
    actions = [
      "s3:ListBucket",
      "s3:GetBucketLocation"
    ]
    resources = [
      "arn:aws:s3:::${var.s3_bucket_name}"
    ]
  }

  statement {
    actions = [
      "s3:GetObject"
    ]
    resources = [
      "arn:aws:s3:::${var.s3_bucket_name}/*"
    ]
  }
}

resource "aws_iam_policy" "ecs_s3_policy" {
  name        = "${var.name_prefix}-ecs-s3-access"
  description = "Allows ECS tasks to list and download snapshots from S3"
  policy      = data.aws_iam_policy_document.ecs_s3_access.json
}

resource "aws_iam_role" "ecs_execution" {
  name               = "${var.name_prefix}-ecs-execution-role"
  assume_role_policy = data.aws_iam_policy_document.ecs_task_assume_role.json
  tags               = local.common_tags
}


resource "aws_iam_role_policy_attachment" "ecs_execution_managed" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "ecs_task" {
  name               = "${var.name_prefix}-ecs-task-role"
  assume_role_policy = data.aws_iam_policy_document.ecs_task_assume_role.json
  tags               = local.common_tags
}

resource "aws_iam_role_policy_attachment" "ecs_s3_attach" {
  role       = aws_iam_role.ecs_task.name
  policy_arn = aws_iam_policy.ecs_s3_policy.arn
}

data "aws_iam_policy_document" "ecs_task_s3_read" {
  count = var.snapshot_bucket_name == null ? 0 : 1

  statement {
    sid = "ReadSnapshotObjects"
    actions = [
      "s3:GetObject",
    ]
    resources = [local.snapshot_objects_arn]
  }

  statement {
    sid = "ListSnapshotPrefix"
    actions = [
      "s3:ListBucket",
    ]
    resources = [local.snapshot_bucket_arn]
  }
}

resource "aws_iam_role_policy" "ecs_task_s3_read" {
  count = var.snapshot_bucket_name == null ? 0 : 1

  name   = "${var.name_prefix}-ecs-task-s3-read"
  role   = aws_iam_role.ecs_task.id
  policy = data.aws_iam_policy_document.ecs_task_s3_read[0].json
}

resource "aws_lb" "this" {
  name               = local.alb_name
  load_balancer_type = "application"
  internal           = false
  security_groups    = [aws_security_group.alb.id]
  subnets            = values(aws_subnet.public)[*].id
  idle_timeout = 3600

  tags = local.common_tags
}

resource "aws_lb_target_group" "ecs" {
  name        = "${var.name_prefix}-tg"
  port        = var.container_port
  protocol    = "HTTP"
  vpc_id      = aws_vpc.this.id
  target_type = "ip"

  health_check {
    path                = var.health_check_path
    healthy_threshold   = 2
    unhealthy_threshold = 3
    interval            = 30
    timeout             = 5
    matcher             = "200-399"
  }

  tags = local.common_tags
}

resource "aws_acm_certificate" "this" {
  domain_name       = var.domain_name
  validation_method = "DNS"

  options {
    certificate_transparency_logging_preference = var.acm_certificate_transparency_logging_preference
  }

  lifecycle {
    create_before_destroy = true
  }

  tags = local.common_tags
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.this.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type = "redirect"
    redirect {
      protocol    = "HTTPS"
      port        = "443"
      status_code = "HTTP_301"
    }
  }
}

resource "aws_lb_listener" "https" {
  load_balancer_arn = aws_lb.this.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = aws_acm_certificate.this.arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.ecs.arn
  }
}

resource "aws_ecs_task_definition" "this" {
  family                   = local.task_family_name
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = tostring(var.task_cpu)
  memory                   = tostring(var.task_memory)
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  volume {
    name = "everspring-cache"
    efs_volume_configuration {
      file_system_id     = aws_efs_file_system.cache.id
      transit_encryption = "ENABLED"
      authorization_config {
        access_point_id = aws_efs_access_point.cache_path.id
        iam             = "DISABLED"
      }
    }
  }

  container_definitions = jsonencode([
    {
      name      = var.container_name
      image     = local.app_image
      essential = true
      command   = var.task_command
      portMappings = [
        {
          containerPort = var.container_port
          hostPort      = var.container_port
          protocol      = "tcp"
        }
      ]
      environment = [
        { name = "EVERSPRING_MODEL_TIER", value = "slim" },
        { name = "EVERSPRING_DATA_DIR", value = "/home/everspring/.everspring" },
        { name = "EVERSPRING_HTTP_HOST", value = "0.0.0.0" },
        { name = "EVERSPRING_HTTP_PORT", value = tostring(var.container_port) }
      ]
      mountPoints = [
        {
          sourceVolume  = "everspring-cache"
          containerPath = "/home/everspring/.everspring"
          readOnly      = false
        }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.ecs.name
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "ecs"
        }
      }
    }
  ])

  depends_on = [aws_efs_mount_target.private]

  tags = local.common_tags
}

resource "aws_ecs_service" "this" {
  name            = local.ecs_service_name
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.this.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  network_configuration {
    subnets          = values(aws_subnet.private)[*].id
    security_groups  = [aws_security_group.ecs.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.ecs.arn
    container_name   = var.container_name
    container_port   = var.container_port
  }

  depends_on = [aws_lb_listener.https]

  tags = local.common_tags
}
