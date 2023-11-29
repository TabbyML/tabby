# Run loadtest with tabby on modal GPUs

Steps:
1. Adjust `TABBY_API_HOST` in `run.sh` to match your modal deployment url.
2. Add models you're interested in to benchmark at end of `run.sh`
3. Run `run.sh`, output will be appended to `record.csv`