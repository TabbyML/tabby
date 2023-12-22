# Skypilot Serving
[SkyPilot](https://skypilot.readthedocs.io/en/latest/) is a framework for running LLMs, AI, and batch jobs on any cloud, offering maximum cost savings, highest GPU availability, and managed execution.

[SkyServe](https://skypilot.readthedocs.io/en/latest/serving/sky-serve.html) is SkyPilot’s model serving library. SkyServe (short for SkyPilot Serving) takes an existing serving framework and deploys it across one or more regions or clouds.

## Configuration

At first, let's specified the resource requirements for the Tabby service in the YAML configuration for SkyServe.

```yaml
resources:
  ports: 8080
  accelerators: T4:1
```

Skypilot supports GPU from various cloud vendors. Please refer to the official [Skypilot documentation](https://skypilot.readthedocs.io/en/latest/getting-started/installation.html) for detailed installation instructions.

As Tabby exposes its health check at `/v1/health`, we can define the following service configuration:

```yaml
service:
  readiness_probe: /v1/health
  replicas: 1
```

Finally, we define the command line that actually initiates the container job:

```yaml
run: |
  docker run --gpus all -p 8080:8080 -v ~/.tabby:/data \
    tabbyml/tabby \
    serve --model TabbyML/StarCoder-1B --device cuda
```

## Launch the service

We first execute `sky serve up tabby.yaml -n tabby`.

![start tabby service](./start-service.png)

If everything goes well, you'll see messages below
![service ready](./service-ready.png)

This only initiates SkyServe's scheduler and load balancer; the actual Tabby service is not yet activated.

When you execute the following command, you'll encounter a message indicating that the replica is not ready:

```bash
$ curl -L 'http://44.203.34.65:30001/v1/health'

{"detail":"No available replicas. Use \"sky serve status [SERVICE_NAME]\" to check the replica status."}%
```

You can monitor the progress of starting the actual tabby job by checking the replica log:

```bash
# Tailing the logs of replica 1 for the tabby service
sky serve logs tabby 1
```

Once the service is ready, you will see something like the following:

![tabby ready](./tabby-ready.png)

SkyServe uses a redirect load balancer at its front, so the `-L` command is necessary if you would like to test the completion api with `curl`.

```bash
$ curl -L -X 'POST' \
  'http://44.203.34.65:30001/v1/completions' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "language": "python",
  "segments": {
    "prefix": "def fib(n):\n    ",
    "suffix": "\n        return fib(n - 1) + fib(n - 2)"
  }
}'

{"id":"cmpl-ba9aae81-ed9c-419b-9616-fceb92cdbe79","choices":[{"index":0,"text":"    if n <= 1:\n            return n"}]}
```

Now, you can utilize the load balancer URL (`http://44.203.34.65:30001` in this case) within Tabby editor extensions. Please refer to `tabby.yaml` for the comprehensive configuration usedin this tutorial.