#!/bin/bash
# Load Testing Script
# Usage: ./scripts/run_load_test.sh [options]

set -e

# Default values
HOST="${HOST:-http://localhost:8001}"
USERS="${USERS:-10}"
SPAWN_RATE="${SPAWN_RATE:-2}"
DURATION="${DURATION:-60s}"

echo "=========================================="
echo "Credit Scoring API Load Test"
echo "=========================================="
echo "Host: $HOST"
echo "Users: $USERS"
echo "Spawn Rate: $SPAWN_RATE users/second"
echo "Duration: $DURATION"
echo "=========================================="
echo ""

# Check if locust is installed
if ! command -v locust &> /dev/null; then
    echo "Error: locust is not installed"
    echo "Install it with: pip install locust"
    exit 1
fi

# Run locust
locust -f tests/load_testing/locustfile.py \
    --host="$HOST" \
    --users="$USERS" \
    --spawn-rate="$SPAWN_RATE" \
    --run-time="$DURATION" \
    --headless \
    --html=load_test_report.html \
    --csv=load_test_results

echo ""
echo "=========================================="
echo "Load test completed!"
echo "Report saved to: load_test_report.html"
echo "CSV results saved to: load_test_results_*.csv"
echo "=========================================="
