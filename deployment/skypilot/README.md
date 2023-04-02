# Run Tabby server on any cloud with one click

## Background

[**SkyPilot**](https://github.com/skypilot-org/skypilot) is an open-source framework for seamlessly running machine learning on any cloud. With a simple CLI, users can easily launch many clusters and jobs, while substantially lowering their cloud bills. Currently, [Lambda Labs](https://skypilot.readthedocs.io/en/latest/getting-started/installation.html#lambda-cloud) (low-cost GPU cloud), [AWS](https://skypilot.readthedocs.io/en/latest/getting-started/installation.html#aws), [GCP](https://skypilot.readthedocs.io/en/latest/getting-started/installation.html#gcp), and [Azure](https://skypilot.readthedocs.io/en/latest/getting-started/installation.html#azure) are supported. See [docs](https://skypilot.readthedocs.io/en/latest/) to learn more.

## Steps

1. Install SkyPilot and [check that cloud credentials exist](https://skypilot.readthedocs.io/en/latest/getting-started/installation.html#cloud-account-setup):
    ```bash
    pip install "skypilot[aws,gcp,azure,lambda]"  # pick your clouds
    sky check
    ```
    <img src="https://i.imgur.com/7BUci5n.png" width="485" alt="`sky check` output showing enabled clouds for SkyPilot"/>

2. Get the [deployment folder](./):
    ```bash
    git clone https://github.com/TabbyML/tabby
    cd tabby/deployment/skypilot
    ```

3. run:
    ```bash
    sky launch -c tabby default.yml
    ```

4. Open another terminal and run:
    ```bash
    ssh -L 8501:localhost:8501 -L 5000:localhost:5000 -L 8080:localhost:8080 tabby
    ```

5. Open http://localhost:8501 in your browser and start coding!
![tabby admin server](https://user-images.githubusercontent.com/388154/227792390-ec19e9b9-ebbb-4a94-99ca-8a142ffb5e46.png)

## Cleaning up
When you are done, you can stop or tear down the cluster:

- **To stop the cluster**, run
    ```bash
    sky stop tabby # or pass your custom name if you used "-c <other name>"
    ```
    You can restart a stopped cluster and relaunch the chatbot (the `run` section in YAML) with
    ```bash
    sky launch default.yml -c tabby --no-setup
    ```
    Note the `--no-setup` flag: a stopped cluster preserves its disk contents so we can skip redoing the setup.
- **To tear down the cluster** (non-restartable), run
    ```bash
    sky down tabby # or pass your custom name if you used "-c <other name>"
    ```
