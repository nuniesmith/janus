#!/bin/bash
# Initialize Docker Secrets for Data Factory
# 
# Usage:
#   Development: ./scripts/init-secrets.sh dev
#   Production:  ./scripts/init-secrets.sh prod

set -e

MODE="${1:-dev}"
SECRETS_DIR="./secrets"

echo "🔐 Initializing Docker Secrets for Data Factory (mode: $MODE)"
echo

# Create secrets directory if it doesn't exist
mkdir -p "$SECRETS_DIR"

# Function to create a secret file
create_secret() {
    local name=$1
    local prompt=$2
    local file="$SECRETS_DIR/${name}.txt"
    
    if [ "$MODE" = "dev" ]; then
        # Development mode: create placeholder
        if [ ! -f "$file" ]; then
            echo "PLACEHOLDER_${name}_$(date +%s)" > "$file"
            echo "✓ Created placeholder: $file"
        else
            echo "⊙ Exists: $file"
        fi
    else
        # Production mode: prompt for actual values
        if [ ! -f "$file" ]; then
            echo -n "$prompt: "
            read -s value
            echo
            if [ -n "$value" ]; then
                echo "$value" > "$file"
                chmod 600 "$file"
                echo "✓ Created: $file"
            else
                echo "⚠ Skipped (empty): $file"
            fi
        else
            echo "⊙ Exists: $file"
        fi
    fi
}

# Exchange API credentials
create_secret "binance_api_key" "Enter Binance API Key"
create_secret "binance_api_secret" "Enter Binance API Secret"
create_secret "bybit_api_key" "Enter Bybit API Key"
create_secret "bybit_api_secret" "Enter Bybit API Secret"
create_secret "kucoin_api_key" "Enter Kucoin API Key"
create_secret "kucoin_api_secret" "Enter Kucoin API Secret"
create_secret "kucoin_api_passphrase" "Enter Kucoin API Passphrase"

# Additional API keys
create_secret "alphavantage_api_key" "Enter AlphaVantage API Key"
create_secret "coinmarketcap_api_key" "Enter CoinMarketCap API Key"

echo
echo "✅ Secrets initialization complete!"
echo
echo "📁 Secrets location: $SECRETS_DIR"
echo

if [ "$MODE" = "dev" ]; then
    echo "⚠️  DEVELOPMENT MODE: Placeholder values created"
    echo "   Replace with real values before production deployment!"
    echo
    echo "   Example:"
    echo "   echo 'your_actual_api_key' > $SECRETS_DIR/binance_api_key.txt"
else
    echo "🔒 PRODUCTION MODE: Real credentials stored"
    echo "   Ensure secrets directory is NOT committed to git!"
    echo "   Add to .gitignore: secrets/*.txt"
fi

echo
echo "🚀 Next steps:"
echo "   1. Review secrets in $SECRETS_DIR"
echo "   2. Start services: docker-compose -f docker-compose.secrets.yml up"
echo

