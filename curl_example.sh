#!/bin/bash


BASE_URL="http://localhost:8080"

echo "Testing train_model endpoint..."
curl -X POST "${BASE_URL}/train_model" \
     -H "Content-Type: application/json"

echo -e "\nTesting test_model endpoint..."
curl -X POST "${BASE_URL}/test_model" \
     -H "Content-Type: application/json" \
     -d '{
        "text": "This is a happy image",
        "image_path": "path/to/your/image.jpg"
     }'