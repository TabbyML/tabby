## Deployment

1. Start service
   ```bash
   docker-compose up
   ```
2. Test API endpoint with curl
   ```bash
   curl -X POST http://localhost:5000/v1/completions -H 'Content-Type: application/json' --data '{
       "prompt": "def binarySearch(arr, left, right, x):\n    mid = (left +"
   }'
   ```
