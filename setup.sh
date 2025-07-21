#!/bin/bash

# Cryptocurrency MLOps System Setup Script
# Author: Mert Göker
# GitHub: https://github.com/MertGoker

set -e

echo "🚀 Welcome to Cryptocurrency MLOps System Setup!"
echo "================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}$1${NC}"
}

# Check if Docker is installed
check_docker() {
    print_status "Checking Docker installation..."
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        print_status "Visit: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        print_status "Visit: https://docs.docker.com/compose/install/"
        exit 1
    fi
    
    print_status "Docker and Docker Compose are installed."
}

# Check if .env file exists
check_env_file() {
    print_status "Checking environment configuration..."
    if [ ! -f .env ]; then
        print_warning ".env file not found. Creating from template..."
        cp env.example .env
        print_status "Please edit .env file with your API keys and configuration."
        print_status "Optional API keys (for enhanced features):"
        echo "  - LANGSMITH_API_KEY (for LLM monitoring and tracing)"
        echo "  - NEWS_API_KEY (for real-time news sentiment analysis)"
        echo ""
        print_status "The system uses free models and can run without any API keys!"
        print_warning "All AI features work with free, open-source models."
    else
        print_status ".env file found."
    fi
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    mkdir -p data/logs
    mkdir -p data/models
    mkdir -p data/cache
    mkdir -p nginx/ssl
    print_status "Directories created successfully."
}

# Build and start services
start_services() {
    print_header "Starting Cryptocurrency MLOps System..."
    echo ""
    
    print_status "Building Docker images..."
    docker-compose build
    
    print_status "Starting services..."
    docker-compose up -d
    
    print_status "Waiting for services to be ready..."
    sleep 30
    
    # Check if services are running
    print_status "Checking service health..."
    
    # Check backend
    if curl -f http://localhost:8000/health &> /dev/null; then
        print_status "✅ Backend API is running"
    else
        print_warning "⚠️  Backend API is not responding"
    fi
    
    # Check frontend
    if curl -f http://localhost:8501/_stcore/health &> /dev/null; then
        print_status "✅ Frontend is running"
    else
        print_warning "⚠️  Frontend is not responding"
    fi
    
    # Check Jenkins
    if curl -f http://localhost:8080 &> /dev/null; then
        print_status "✅ Jenkins CI/CD is running"
    else
        print_warning "⚠️  Jenkins is not responding"
    fi
}

# Display access information
show_access_info() {
    echo ""
    print_header "🎉 Cryptocurrency MLOps System is ready!"
    echo ""
    echo "Access URLs:"
    echo "  🌐 Frontend (Streamlit): http://localhost:8501"
    echo "  🔧 Backend API: http://localhost:8000"
    echo "  📚 API Documentation: http://localhost:8000/docs"
    echo "  🔄 Jenkins CI/CD: http://localhost:8080"
    echo "  🗄️  PostgreSQL: localhost:5432"
    echo "  🔴 Redis: localhost:6379"
    echo ""
    echo "Default credentials:"
    echo "  PostgreSQL: mlops_user / mlops_password"
    echo "  Jenkins: Check logs for initial admin password"
    echo ""
    print_status "To stop the system: docker-compose down"
    print_status "To view logs: docker-compose logs -f"
    echo ""
}

# Show developer information
show_developer_info() {
    print_header "👨‍💻 Developer Information"
    echo ""
    echo "Developer: Mert Göker"
    echo "Role: Data Scientist & ML Engineer"
    echo ""
    echo "Contact:"
    echo "  📧 Email: mert.goker.work@gmail.com"
    echo "  💼 LinkedIn: https://www.linkedin.com/in/mert-goker-bb4bb91b6/"
    echo "  📱 GitHub: https://github.com/MertGoker"
    echo ""
    echo "This project demonstrates expertise in:"
    echo "  - End-to-end ML system development"
    echo "  - MLOps and CI/CD pipelines"
    echo "  - Free LLM integration and RAG pipelines"
    echo "  - Real-time data processing"
    echo "  - Docker containerization"
    echo "  - Open-source AI model deployment"
    echo ""
}

# Main setup function
main() {
    print_header "Cryptocurrency MLOps System Setup"
    echo "========================================"
    echo ""
    
    check_docker
    check_env_file
    create_directories
    start_services
    show_access_info
    show_developer_info
    
    echo ""
    print_status "Setup completed successfully!"
    print_status "Happy trading! 📈"
}

# Run main function
main "$@" 