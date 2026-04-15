pipeline {
    agent any
    
    parameters {
        choice(name: 'TIER', choices: ['main', 'slim', 'xslim'], description: 'Application Worker Model (Tier)')
        string(name: 'WORKERS', defaultValue: '1', description: 'Number of workers to run')
        string(name: 'BACKLOG', defaultValue: '2048', description: 'Socket backlog size')
        string(name: 'THREADS', defaultValue: '1', description: 'Number of threads per worker')
    }

    environment {
        // Container name defaults locally; registry/auth values must be explicitly provided by Jenkins job configuration.
        CONTAINER_NAME = "${env.CONTAINER_NAME ?: 'everspring-mcp-app'}"
    }

    stages {
        stage('Cleanup & Login') {
            steps {
                script {
                    if (!env.IMAGE_NAME?.trim()) {
                        error('Required Jenkins job variable IMAGE_NAME is not configured (set it in Jenkins job configuration/environment).')
                    }
                    if (!env.GHCR_CREDENTIALS_ID?.trim()) {
                        error('Required Jenkins job variable GHCR_CREDENTIALS_ID is not configured (set it in Jenkins job configuration/environment).')
                    }
                    if (!env.GHCR_USERNAME?.trim()) {
                        error('Required Jenkins job variable GHCR_USERNAME is not configured (set it in Jenkins job configuration/environment).')
                    }
                }
                withCredentials([string(credentialsId: env.GHCR_CREDENTIALS_ID, variable: 'GHCR_PAT')]) {
                    sh """
                        export XDG_RUNTIME_DIR=/run/user/\$(id -u)
                        echo \$GHCR_PAT | podman login ghcr.io -u \$GHCR_USERNAME --password-stdin
                    """
                }
            }
        }
        
        stage('Pull Image') {
            steps {
                sh """
                    export XDG_RUNTIME_DIR=/run/user/\$(id -u)
                    podman pull ${IMAGE_NAME}:latest
                """
            }
        }

        stage('Deploy') {
            steps {
                sh """
                    export XDG_RUNTIME_DIR=/run/user/\$(id -u)
                    
                    # Clear existing container if it exists
                    podman stop ${CONTAINER_NAME} || true
                    podman rm ${CONTAINER_NAME} || true
                    
                    podman run -d \
                        --name ${CONTAINER_NAME} \
                        --restart always \
                        --memory="4g" \
                        --memory-swap="6g" \
                        --cpus="2.0" \
                        -p 8000:8000 \
                        -v everspring-data-${params.TIER}:/home/everspring/.everspring \
                        ${IMAGE_NAME}:latest \
                        python -m everspring_mcp.main serve \
                            --tier ${params.TIER} \
                            --transport http \
                            --workers ${params.WORKERS} \
                            --backlog ${params.BACKLOG} \
                            --threads ${params.THREADS}
                """
            }
        }
    }
}
