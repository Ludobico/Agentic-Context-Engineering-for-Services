#!/bin/bash

set -e

echo "Starting ACE Framework Container..."

if [ ! -f /app/config/config.ini ]; then
    echo "⚠️ config.ini not found. Creating from example..."
    cp /app/config/config-example.ini /app/config/config.ini
else
    echo "config.ini found."
fi

# 원래 실행하려던 명령어 실행 (python main.py 등)
exec "$@"