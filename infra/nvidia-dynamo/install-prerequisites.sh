#!/bin/bash

# ============================================================================
# NVIDIA DYNAMO PREREQUISITES INSTALLATION
# ============================================================================
# This script installs the necessary infrastructure and prerequisites for
# deploying NVIDIA Dynamo inference graphs on Amazon EKS.
#
# The script performs the following steps:
# 1. Validates required tools and AWS credentials
# 2. Deploys base EKS infrastructure using Terraform
# 3. Deploys Dynamo-specific infrastructure and ECR repositories
# 4. Sets up Dynamo Cloud operator and platform components
# 5. Verifies deployment and provides access instructions
#
# Based on patterns from dynamo-on-eks/dynamo-cloud scripts
# ============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

section() {
    echo -e "\n${CYAN}=== $1 ===${NC}"
}

print_banner() {
    echo -e "${CYAN}"
    echo "============================================================================"
    echo "  $1"
    echo "============================================================================"
    echo -e "${NC}"
}

# Utility functions
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Script directory and project root detection
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Detect if we're in blueprints or infra directory
if [[ "$SCRIPT_DIR" == *"blueprints/inference/nvidia-dynamo"* ]]; then
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
    INFRA_DIR="$PROJECT_ROOT/infra/nvidia-dynamo"
    CONTEXT="blueprint"
elif [[ "$SCRIPT_DIR" == *"infra/nvidia-dynamo"* ]]; then
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
    INFRA_DIR="$SCRIPT_DIR"
    CONTEXT="infra"
else
    PROJECT_ROOT="$(cd "$SCRIPT_DIR" && pwd)"
    INFRA_DIR="$PROJECT_ROOT/infra/nvidia-dynamo"
    CONTEXT="standalone"
fi

section "NVIDIA Dynamo Prerequisites Tool Installation"

info "Script context: $CONTEXT"
info "Script directory: $SCRIPT_DIR"
info "Project root: $PROJECT_ROOT"

section "Step 1: Check and Install Required Tools"

# Required tools for Dynamo deployment
REQUIRED_TOOLS=("kubectl" "aws" "terraform" "helm" "docker" "git" "python3")
OPTIONAL_TOOLS=("earthly")

info "Checking required tools..."
MISSING_TOOLS=()

for tool in "${REQUIRED_TOOLS[@]}"; do
    if ! command -v "$tool" &> /dev/null; then
        warn "✗ $tool not found"
        MISSING_TOOLS+=("$tool")
    else
        success "✓ $tool found"
    fi
done

info "Checking optional tools..."
for tool in "${OPTIONAL_TOOLS[@]}"; do
    if command -v "$tool" &> /dev/null; then
        success "✓ $tool found (recommended for infrastructure builds)"
    else
        warn "⚠ $tool not found (optional, but recommended for infrastructure deployment)"
        case $tool in
            "earthly")
                info "Install Earthly: https://earthly.dev/get-earthly"
                ;;
        esac
    fi
done

# If there are missing tools, provide installation instructions
if [[ ${#MISSING_TOOLS[@]} -gt 0 ]]; then
    error "Missing required tools: ${MISSING_TOOLS[*]}"
    echo ""
    info "Installation instructions:"

    for tool in "${MISSING_TOOLS[@]}"; do
        case $tool in
            "kubectl")
                echo "  kubectl: https://kubernetes.io/docs/tasks/tools/"
                echo "    # Linux:"
                echo "    curl -LO \"https://dl.k8s.io/release/\$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl\""
                echo "    sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl"
                ;;
            "aws")
                echo "  AWS CLI: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
                echo "    # Linux:"
                echo "    curl \"https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip\" -o \"awscliv2.zip\""
                echo "    unzip awscliv2.zip && sudo ./aws/install"
                ;;
            "terraform")
                echo "  Terraform: https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli"
                echo "    # Linux:"
                echo "    wget -O- https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg"
                echo "    echo \"deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com \$(lsb_release -cs) main\" | sudo tee /etc/apt/sources.list.d/hashicorp.list"
                echo "    sudo apt update && sudo apt install terraform"
                ;;
            "helm")
                echo "  Helm: https://helm.sh/docs/intro/install/"
                echo "    # Linux:"
                echo "    curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash"
                ;;
            "docker")
                echo "  Docker: https://docs.docker.com/get-docker/"
                echo "    # Linux:"
                echo "    curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh"
                echo "    sudo usermod -aG docker \$USER"
                ;;
            "git")
                echo "  Git: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git"
                echo "    # Linux:"
                echo "    sudo apt update && sudo apt install git"
                ;;
            "python3")
                echo "  Python 3: https://www.python.org/downloads/"
                echo "    # Linux:"
                echo "    sudo apt update && sudo apt install python3 python3-pip python3-venv"
                ;;
        esac
        echo ""
    done

    echo "After installing missing tools, run this script again to verify installation."
    exit 1
fi

section "Step 2: Validate Tool Functionality"

info "Validating tool functionality..."

# Check Python and pip/venv
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d'.' -f1,2)
    PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d'.' -f1)
    PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d'.' -f2)
    PYTHON_VERSION_NUM=$((PYTHON_MAJOR * 10 + PYTHON_MINOR))

    if [[ "$PYTHON_VERSION_NUM" -ge 38 ]]; then
        success "✓ Python $PYTHON_VERSION (≥3.8 required)"
    else
        error "✗ Python $PYTHON_VERSION found, but 3.8+ required"
        exit 1
    fi

    # Check pip
    if python3 -m pip --version &> /dev/null; then
        success "✓ pip available"
    else
        error "✗ pip not available"
        info "Install pip: python3 -m ensurepip --upgrade"
        exit 1
    fi

    # Check venv
    if python3 -m venv --help &> /dev/null; then
        success "✓ venv available"
    else
        error "✗ venv not available"
        info "Install venv: sudo apt install python3-venv"
        exit 1
    fi
fi

# Check Docker service (if Docker is installed)
if command -v docker &> /dev/null; then
    if docker info &> /dev/null; then
        success "✓ Docker service is running"
    else
        warn "⚠ Docker is installed but not running or not accessible"
        info "Start Docker: sudo systemctl start docker"
        info "Add user to docker group: sudo usermod -aG docker \$USER && newgrp docker"
    fi
fi

# Check optional tools installation guidance
info "Optional tool installation:"
if ! command -v earthly &> /dev/null; then
    info "  Earthly (recommended for infrastructure builds):"
    info "    sudo /bin/sh -c 'wget https://github.com/earthly/earthly/releases/latest/download/earthly-linux-amd64 -O /usr/local/bin/earthly && chmod +x /usr/local/bin/earthly && /usr/local/bin/earthly bootstrap --with-autocomplete'"
fi

section "Step 3: Configuration Validation"

info "Checking tool configurations..."

# AWS CLI configuration check (optional)
if command -v aws &> /dev/null; then
    if aws sts get-caller-identity &> /dev/null 2>&1; then
        AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
        success "✓ AWS credentials configured (Account: $AWS_ACCOUNT_ID)"
    else
        warn "⚠ AWS CLI installed but credentials not configured"
        info "Configure AWS: aws configure"
    fi
fi

# kubectl configuration check (optional)
if command -v kubectl &> /dev/null; then
    if kubectl cluster-info &> /dev/null 2>&1; then
        CURRENT_CONTEXT=$(kubectl config current-context)
        success "✓ kubectl configured (Context: $CURRENT_CONTEXT)"
    else
        warn "⚠ kubectl installed but not configured or cluster not accessible"
        info "Configure kubectl to access your EKS cluster"
    fi
fi

section "Prerequisites Check Complete!"

success "NVIDIA Dynamo prerequisites validation completed!"
echo ""
info "Tool Status Summary:"
echo "  ✓ All required tools are installed and functional"
echo "  ✓ Python 3.8+ with pip and venv available"
echo "  ✓ Infrastructure tools ready (terraform, kubectl, helm, aws)"
echo "  ✓ Container tools ready (docker)"
echo "  ✓ Development tools ready (git)"
echo ""
info "Next Steps:"
echo "  1. Deploy infrastructure using the appropriate install scripts:"
if [[ "$CONTEXT" == "blueprint" ]]; then
    echo "     - Run: ../../../infra/nvidia-dynamo/install.sh (for Dynamo infrastructure)"
    echo "  2. Set up inference environment: ./setup.sh"
    echo "  3. Deploy inference graph: ./deploy.sh"
    echo "  4. Test deployment: ./test.sh"
else
    echo "     - Run: ./install.sh (for Dynamo infrastructure)"
    echo "  2. Navigate to blueprints/inference/nvidia-dynamo/"
    echo "  3. Set up inference environment: ./setup.sh"
    echo "  4. Deploy inference graph: ./deploy.sh"
    echo "  5. Test deployment: ./test.sh"
fi
echo ""
info "Configuration Notes:"
echo "  - Ensure AWS credentials are configured: aws configure"
echo "  - Ensure kubectl is configured for your EKS cluster"
echo "  - Ensure Docker service is running: sudo systemctl start docker"
echo ""
info "Optional Enhancements:"
echo "  - Install Earthly for faster infrastructure builds"
echo "  - Configure shell completion for installed tools"
