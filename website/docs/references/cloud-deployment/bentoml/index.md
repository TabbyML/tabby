---
image: ./twitter-img.png
---

# BentoCloud
[BentoCloud](https://cloud.bentoml.com/) provides a serverless infrastructure tailored for GPU workloads, enabling seamless deployment, management, and scaling of models in the cloud.

## Setup

Begin by crafting a `service.py` to delineate your Bento service. This script delineates the GPU resources requisite for operating your service.

```python title="service.py"
@bentoml.service(
    resources={"gpu": 1, "gpu_type": "nvidia-l4"},
    traffic={"timeout": 10},
)
```

BentoCloud currently supports the following GPUs:

- `T4`: A cost-effective GPU selection with 16GiB of memory.
- `L4`: A mid-range GPU offering 24GiB of memory.
- `A100`: The pinnacle of GPU power in the cloud, available in configurations of 40GiB and 80GiB memory options.

For comprehensive details, please refer to the official [BentoCloud Pricing](https://www.bentoml.com/pricing).

## Define the Container Image

To construct a container image replete with the preloaded Tabby model cache, draft a `bentofile.yaml`. This document stipulates the CUDA version as 11.7.1 and enumerates the essential system packages and dependencies for the image. Leveraging BentoCloud's internal filesystem circumvents the need to redownload the model, thereby accelerating cold start times.

Below is the `bentofile.yaml`:

```yaml title="bentofile.yaml"
service: 'service:Tabby'
include:
    - '*.py'
python:
    packages:
        - asgi-proxy-lib 
docker:
    cuda_version: "11.7.1"
    system_packages:
        - unzip
        - git
        - curl
        - software-properties-common
    setup_script: "./setup-docker.sh"
```

The `asgi-proxy-lib` package is specified to facilitate communication with the Tabby server via localhost, and the `setup-docker.sh` script is configured to install Tabby and procure the model weights.

```bash title="setup-docker.sh"
# Install Tabby
DISTRO=tabby_x86_64-manylinux2014-cuda117
curl -L https://github.com/TabbyML/tabby/releases/download/v0.14.0/$DISTRO.zip \
  -o $DISTRO.zip
unzip $DISTRO.zip

# Download model weights under the bentoml user, as BentoCloud operates under this user.
su bentoml -c "TABBY_MODEL_CACHE_ROOT=/home/bentoml/tabby-models tabby download --model StarCoder-1B"
su bentoml -c "TABBY_MODEL_CACHE_ROOT=/home/bentoml/tabby-models tabby download --model Qwen2-1.5B-Instruct"
su bentoml -c "TABBY_MODEL_CACHE_ROOT=/home/bentoml/tabby-models tabby download --model Nomic-Embed-Text"
```

### Service Definition

The service endpoint is encapsulated with BentoML's `@bentoml.service`. Here, we:

1. Initiate the Tabby process and ensure its readiness to process incoming requests.
2. Establish an ASGI proxy to relay requests from the Modal web endpoint to the local Tabby server.
3. Allocate 1 Nvidia L4 GPU per worker, with a 10-second timeout.
4. Employ `on_deployment` and `on_shutdown` hooks to transfer persisted data to and from object storage.

```python title="service.py"
app = asgi_proxy("http://127.0.0.1:8000")

@bentoml.service(
    resources={"gpu": 1, "gpu_type": "nvidia-l4"},
    traffic={"timeout": 10},
)
@bentoml.mount_asgi_app(app, path="/")
class Tabby:
    @bentoml.on_deployment
    def prepare():
        download_tabby_dir("tabby-local")

    @bentoml.on_shutdown
    def shutdown(self):
        upload_tabby_dir("tabby-local")

    def __init__(self) -> None:
        model_id = "StarCoder-1B"
        chat_model_id = "Qwen2-1.5B-Instruct"

        # Fire up the server subprocess.
        self.server = TabbyServer(model_id, chat_model_id)

        # Await server readiness.
        self.server.wait_until_ready()
```

Finally, we draft a deployment configuration file `bentodeploy.yaml` to outline the deployment specifics. Note that we employ rclone to synchronize persisted data with Cloudflare R2 object storage. You can get the values of the following R2 environment variables by referring to the [Cloudfare R2 documentation](https://developers.cloudflare.com/r2/api/s3/tokens/).

```yaml title="bentodeploy.yaml"
name: tabby-local
bento: ./
access_authorization: false
envs:
  - name: RCLONE_CONFIG_R2_TYPE
    value: s3
  - name: RCLONE_CONFIG_R2_ACCESS_KEY_ID
    value: $YOUR_R2_ACCESS_KEY_ID
  - name: RCLONE_CONFIG_R2_SECRET_ACCESS_KEY
    value: $YOUR_R2_SECRET_ACCESS_KEY
  - name: RCLONE_CONFIG_R2_ENDPOINT
    value: $YOUR_R2_ENDPOINT
  - name: TABBY_MODEL_CACHE_ROOT
    value: /home/bentoml/tabby-models
```

### Serve the Application

Deploying the model with `bentoml deploy -f bentodeploy.yaml` will establish a BentoCloud deployment and serve your application.

![app-running](./app-running.png)

Once the deployment is operational, you can access the service via the provided URL, e.g., `https://$YOUR_DEPLOYMENT_SLUG.mt-guc1.bentoml.ai`.

For the complete code of this tutorial, please refer to the [GitHub repository](https://github.com/TabbyML/tabby/tree/main/website/docs/references/cloud-deployment/bentoml).
