#!/bin/bash
# Data Fetching Daemon Launcher
# This script runs the data fetching daemon in the background

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$PROJECT_ROOT/src/data_fetching/main.py"
LOG_DIR="$PROJECT_ROOT/logs"
PID_FILE="$PROJECT_ROOT/data_fetching_daemon.pid"
LOG_FILE="$LOG_DIR/data_fetching_daemon.log"

# Create log directory
mkdir -p "$LOG_DIR"

# Function to check if daemon is running
is_running() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            return 0  # Running
        else
            rm -f "$PID_FILE"
        fi
    fi
    return 1  # Not running
}

# Function to start daemon
start_daemon() {
    if is_running; then
        echo "❌ Daemon is already running (PID: $(cat "$PID_FILE"))"
        exit 1
    fi

    echo "🚀 Starting data fetching daemon..."
    cd "$PROJECT_ROOT"

    # Start daemon in background
    nohup python3 -m src.data_fetching.main --daemon \
         --delay 300 \
         >> "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"

    echo "✅ Daemon started (PID: $(cat "$PID_FILE"))"
    echo "📝 Logs: $LOG_FILE"
    echo "🛑 To stop: $0 stop"
}

# Function to stop daemon
stop_daemon() {
    if ! is_running; then
        echo "❌ Daemon is not running"
        exit 1
    fi

    PID=$(cat "$PID_FILE")
    echo "🛑 Stopping daemon (PID: $PID)..."

    # Send SIGTERM for graceful shutdown
    kill -TERM "$PID"

    # Wait for shutdown
    for i in {1..30}; do
        if ! kill -0 "$PID" 2>/dev/null; then
            rm -f "$PID_FILE"
            echo "✅ Daemon stopped gracefully"

            echo "💡 Dollar bars are automatically generated during runtime"
            exit 0
        fi
        sleep 1
    done

    # Force kill if still running
    echo "⚠️  Daemon didn't stop gracefully, force killing..."
    kill -KILL "$PID" 2>/dev/null || true
    rm -f "$PID_FILE"
    echo "✅ Daemon force-killed"
}

# Function to consolidate datasets
consolidate_datasets() {
    echo "🔄 Consolidating partitioned datasets..."
    if python3 "$PROJECT_ROOT/scripts/consolidate_datasets.py" "$@"; then
        echo "✅ Datasets consolidated successfully"
    else
        echo "❌ Consolidation failed"
        exit 1
    fi
}

# Function to show status
show_status() {
    if is_running; then
        PID=$(cat "$PID_FILE")
        echo "✅ Daemon is running (PID: $PID)"
        echo "📝 Logs: $LOG_FILE"
        echo "📊 Recent log entries:"
        tail -10 "$LOG_FILE" 2>/dev/null || echo "No logs yet"
    else
        echo "❌ Daemon is not running"
        if [ -f "$LOG_FILE" ]; then
            echo "📝 Last logs:"
            tail -5 "$LOG_FILE" 2>/dev/null
        fi
    fi
}

# Main command handling
case "${1:-status}" in
    start)
        start_daemon
        ;;
    stop)
        stop_daemon
        ;;
    restart)
        if is_running; then
            stop_daemon
            sleep 2
        fi
        start_daemon
        ;;
    consolidate)
        shift  # Remove 'consolidate' from arguments
        consolidate_datasets "$@"
        ;;
    base)
        # Use existing consolidated file as base for new partitions
        consolidated_file="$PROJECT_ROOT/data/raw/dataset_raw_final.parquet"
        if [ ! -f "$consolidated_file" ]; then
            echo "❌ No consolidated file found: $consolidated_file"
            echo "Run './scripts/run_data_fetching_daemon.sh consolidate' first"
            exit 1
        fi

        echo "🔄 Setting up base partition from consolidated file..."
        mkdir -p "$PROJECT_ROOT/data/raw/dataset_raw.parquet"
        cp "$consolidated_file" "$PROJECT_ROOT/data/raw/dataset_raw.parquet/part-00000.parquet"
        echo "✅ Base partition created: part-00000.parquet"
        echo "🚀 You can now run './scripts/run_data_fetching_daemon.sh start'"
        ;;
    status)
        show_status
        ;;
    logs)
        if [ -f "$LOG_FILE" ]; then
            tail -f "$LOG_FILE"
        else
            echo "No log file found: $LOG_FILE"
        fi
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|base|status|logs}"
        echo ""
        echo "Commands:"
        echo "  start       - Start the daemon in background"
        echo "  stop        - Stop the daemon gracefully"
        echo "  restart     - Restart the daemon"
        echo "  base        - Use consolidated file as base for new partitions"
        echo "  status      - Show daemon status and recent logs"
        echo "  logs        - Tail the log file"
        echo ""
        echo "Examples:"
        echo "  $0 consolidate --output data/raw/final_dataset.parquet"
        echo "  $0 base  # Use existing consolidated file as base"
        echo "  $0 start && sleep 3600 && $0 stop  # Run for 1 hour then consolidate"
        echo ""
        echo "The daemon fetches data every 5 minutes automatically."
        exit 1
        ;;
esac
