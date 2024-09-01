# SkyPilot Serving

[SkyPilot](https://skypilot.readthedocs.io/en/latest/) is a versatile framework designed for the execution of LLMs, AI, and batch jobs on any cloud vendors. It stands out by offering significant cost savings, optimal GPU availability, and managed execution capabilities.

[SkyServe](https://skypilot.readthedocs.io/en/latest/serving/sky-serve.html) is SkyPilot’s model serving library. SkyServe (short for SkyPilot Serving) takes an existing serving framework and deploys it across one or more regions or clouds.

When leveraging SkyServe, all replica Tabby instances are seamlessly deployed within your own cloud accounts and VPCs.

## Configuration

At first, let's specified the resource requirements for the Tabby service in the YAML configuration for SkyServe.

```yaml
resources:
  ports: 8080
  accelerators: T4:1
  # Or, allow using any of these GPUs to enhance GPU availability.
  # SkyPilot will auto-select the cheapest and available GPU.
  # accelerators: {T4:1, L4:1, A100:1, A10G:1}
```

Skypilot supports GPU from various cloud vendors. Please refer to the official [Skypilot documentation](https://skypilot.readthedocs.io/en/latest/getting-started/installation.html) for detailed installation instructions.

Tabby exposes its health check at the `/metrics` endpoint, which also serves as a prometrics endpoint. Therefore, we can define the following readiness probe:

```yaml
service:
  readiness_probe: /metrics
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

This finishes launching SkyServe's control VM which runs a load balancer for this serve; the actual replica running the Tabby service is undergoing provisioning.

When you execute the following command, you'll encounter a message indicating that the replica is not ready:

```bash
$ curl -L 'http://44.203.34.65:30001/metrics'

{"detail":"No available replicas. Use \"sky serve status [SERVICE_NAME]\" to check the replica status."}%
```

You can monitor the progress of starting the actual tabby job by checking the replica log:

```bash
# Tailing the logs of replica 1 for the tabby service
sky serve logs tabby 1
```

Once the service is ready, you will see something like the following:

![tabby ready](./tabby-ready.png)

Now, you can utilize the load balancer URL (`http://44.203.34.65:30001` in this case) within Tabby editor extensions. Please refer to [`tabby.yaml`](https://github.com/TabbyML/tabby/blob/main/website/docs/installation/skypilot/tabby.yaml) for the full configuration used in this tutorial.
