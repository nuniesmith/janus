#!/bin/bash
# ============================================================================
# JANUS Data Service - Quick Start Script
# ============================================================================
# This script helps you get the data service running in minutes.
#
# Usage:
#   ./quick-start.sh          # Interactive setup
#   ./quick-start.sh --auto   # Auto-start with defaults
#   ./quick-start.sh --clean  # Clean everything and start fresh

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}  $1"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_step() {
    echo -e "${BLUE}▶${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

prompt_yes_no() {
    local prompt="$1"
    local default="${2:-n}"

    if [ "$AUTO_MODE" = true ]; then
        [ "$default" = "y" ] && return 0 || return 1
    fi

    while true; do
        read -p "$prompt [y/n] (default: $default): " yn
        yn=${yn:-$default}
        case $yn in
            [Yy]* ) return 0;;
            [Nn]* ) return 1;;
            * ) echo "Please answer yes or no.";;
        esac
    done
}

# ============================================================================
# Parse Arguments
# ============================================================================

AUTO_MODE=false
CLEAN_MODE=false

for arg in "$@"; do
    case $arg in
        --auto)
            AUTO_MODE=true
            shift
            ;;
        --clean)
            CLEAN_MODE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --auto    Auto-start with defaults (no prompts)"
            echo "  --clean   Clean everything and start fresh"
            echo "  --help    Show this help message"
            exit 0
            ;;
    esac
done

# ============================================================================
# Main Script
# ============================================================================

print_header "JANUS Data Service - Quick Start"

# Check if we're in the right directory
if [ ! -f "Cargo.toml" ]; then
    print_error "Please run this script from the data service directory:"
    echo "  cd src/janus/services/data"
    exit 1
fi

# ============================================================================
# Step 1: Clean Mode (optional)
# ============================================================================

if [ "$CLEAN_MODE" = true ]; then
    print_header "Cleaning Previous Installation"

    print_step "Stopping Docker containers..."
    docker-compose down -v 2>/dev/null || true
    print_success "Containers stopped"

    print_step "Cleaning build artifacts..."
    cargo clean
    print_success "Build artifacts cleaned"

    print_step "Removing .env file..."
    rm -f .env
    print_success "Environment file removed"

    print_warning "Clean complete. Proceeding with fresh setup..."
    sleep 2
fi

# ============================================================================
# Step 2: Check Prerequisites
# ============================================================================

print_header "Step 1: Checking Prerequisites"

# Check Rust
print_step "Checking Rust installation..."
if ! command -v cargo &> /dev/null; then
    print_error "Rust is not installed!"
    echo "Install from: https://rustup.rs/"
    exit 1
fi
RUST_VERSION=$(rustc --version | cut -d' ' -f2)
print_success "Rust $RUST_VERSION installed"

# Check Docker
print_step "Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed!"
    echo "Install from: https://docs.docker.com/get-docker/"
    exit 1
fi
print_success "Docker $(docker --version | cut -d' ' -f3 | tr -d ',') installed"

# Check Docker Compose
print_step "Checking Docker Compose..."
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    print_error "Docker Compose is not installed!"
    exit 1
fi
print_success "Docker Compose available"

# ============================================================================
# Step 3: Set Up Environment
# ============================================================================

print_header "Step 2: Environment Configuration"

if [ ! -f ".env" ]; then
    print_step "Creating .env file from template..."

    if [ -f ".env.starter" ]; then
        cp .env.starter .env
        print_success ".env file created"
    else
        print_error ".env.starter not found!"
        exit 1
    fi

    if [ "$AUTO_MODE" = false ]; then
        if prompt_yes_no "Do you want to edit the .env file now?" "n"; then
            ${EDITOR:-nano} .env
        fi
    fi
else
    print_warning ".env file already exists, skipping..."
fi

# ============================================================================
# Step 4: Start Infrastructure
# ============================================================================

print_header "Step 3: Starting Infrastructure Services"

# Go to project root to find docker-compose.yml
cd ../../../../..

print_step "Starting Redis..."
docker-compose up -d redis
sleep 2
print_success "Redis started on port 6379"

print_step "Starting QuestDB..."
docker-compose up -d questdb
sleep 3
print_success "QuestDB started on ports 9000 (HTTP) and 9009 (ILP)"

print_step "Verifying services..."

# Check Redis
if docker ps | grep -q redis; then
    print_success "Redis is running"
else
    print_error "Redis failed to start"
    exit 1
fi

# Check QuestDB
if docker ps | grep -q questdb; then
    print_success "QuestDB is running"
else
    print_error "QuestDB failed to start"
    exit 1
fi

# Wait for QuestDB to be ready
print_step "Waiting for QuestDB to be ready..."
MAX_RETRIES=30
RETRY=0
while [ $RETRY -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:9000 > /dev/null 2>&1; then
        print_success "QuestDB is ready"
        break
    fi
    RETRY=$((RETRY+1))
    sleep 1
    echo -n "."
done
echo ""

if [ $RETRY -eq $MAX_RETRIES ]; then
    print_warning "QuestDB might not be fully ready, but continuing..."
fi

# Back to data service directory
cd src/janus/services/data

# ============================================================================
# Step 5: Create QuestDB Tables
# ============================================================================

print_header "Step 4: Creating Database Schema"

print_step "Creating trades table in QuestDB..."

TRADES_TABLE_SQL="CREATE TABLE IF NOT EXISTS trades (
    timestamp TIMESTAMP,
    symbol SYMBOL capacity 256 CACHE,
    exchange SYMBOL capacity 16 CACHE,
    side SYMBOL capacity 2 CACHE,
    price DOUBLE,
    amount DOUBLE,
    trade_id SYMBOL capacity 128 CACHE,
    exchange_ts TIMESTAMP,
    receipt_ts TIMESTAMP
) TIMESTAMP(timestamp) PARTITION BY DAY WAL;"

if curl -G "http://localhost:9000/exec" \
    --data-urlencode "query=$TRADES_TABLE_SQL" \
    -s -o /dev/null -w "%{http_code}" | grep -q "200"; then
    print_success "Trades table created"
else
    print_warning "Could not create trades table (may already exist)"
fi

# Create candles table
print_step "Creating candles table..."

CANDLES_TABLE_SQL="CREATE TABLE IF NOT EXISTS candles (
    timestamp TIMESTAMP,
    symbol SYMBOL capacity 256 CACHE,
    exchange SYMBOL capacity 16 CACHE,
    timeframe SYMBOL capacity 8 CACHE,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume DOUBLE,
    trades_count LONG
) TIMESTAMP(timestamp) PARTITION BY DAY WAL;"

if curl -G "http://localhost:9000/exec" \
    --data-urlencode "query=$CANDLES_TABLE_SQL" \
    -s -o /dev/null -w "%{http_code}" | grep -q "200"; then
    print_success "Candles table created"
else
    print_warning "Could not create candles table (may already exist)"
fi

# ============================================================================
# Step 6: Build the Service
# ============================================================================

print_header "Step 5: Building Data Service"

print_step "Compiling Rust code (this may take a few minutes)..."

if cargo build --release 2>&1 | tee /tmp/cargo_build.log; then
    print_success "Build completed successfully"
else
    print_error "Build failed. Check /tmp/cargo_build.log for details"
    exit 1
fi

# ============================================================================
# Step 7: Run the Service
# ============================================================================

print_header "Step 6: Starting Data Service"

print_success "Setup complete! 🎉"
echo ""
print_step "To start the data service, run:"
echo ""
echo "  cargo run --release --bin janus-data"
echo ""
print_step "Or run in the background:"
echo ""
echo "  nohup cargo run --release --bin janus-data > logs/data.log 2>&1 &"
echo ""
print_step "Useful endpoints:"
echo ""
echo "  Health:     http://localhost:8080/health"
echo "  Metrics:    http://localhost:8080/metrics"
echo "  QuestDB UI: http://localhost:9000"
echo ""
print_step "To query trades in QuestDB:"
echo ""
echo "  SELECT * FROM trades ORDER BY timestamp DESC LIMIT 100;"
echo ""
print_step "To stop services:"
echo ""
echo "  docker-compose down"
echo ""

if prompt_yes_no "Do you want to start the data service now?" "y"; then
    print_step "Starting data service..."
    echo ""
    print_warning "Press Ctrl+C to stop the service"
    echo ""
    sleep 2

    # Create logs directory if it doesn't exist
    mkdir -p logs

    # Run the service
    cargo run --release --bin janus-data
else
    print_success "Setup complete! Run the service manually when ready."
fi
