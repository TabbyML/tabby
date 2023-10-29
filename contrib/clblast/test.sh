curl -L 'http://127.0.0.1:8080/v1/completions' \
-H 'Content-Type: application/json' \
-H 'Accept: application/json' \
-d '{
  "language": "python",
  "segments": {
    "prefix": "def fib(n):\n    ",
    "suffix": "\n        return fib(n - 1) + fib(n - 2)"
  }
}'
