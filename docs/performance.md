# Performance

## Measuring performance

### Throughput

The command line option `--log_throughput` reports *tokens generated per second* on the standard error output. This is the recommended metric to compare different runs (higher is better).

### Profiling

The command line option `--log_profiling` reports an execution profile on the standard error output. It prints a list of selected functions in the format:

```text
  2.51%  80.38%  87.27% beam_search                 557.00ms
```

where the columns mean:

1. Percent of time spent in the function
2. Percent of time spent in the function and its callees
3. Percent of time printed so far
4. Name of the function
5. Time spent in the function (in milliseconds)

The list is ordered on 5. from the largest to smallest time.
