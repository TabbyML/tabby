---
title: Enhanced Tabby Deployment on Modal
authors:
  - name: moqimoqidea
    url: https://github.com/moqimoqidea
    image_url: https://github.com/moqimoqidea
tags: [deployment]
---

# Enhanced Tabby Deployment on Modal: Utilizing Persistent Volumes and Model Caching

In this post, we delve into recent enhancements to Tabby's deployment on Modal, focusing on model caching and the use of persistent volumes. These upgrades significantly improve both scalability and usability within serverless environments.

## Understanding Model Caching

Model caching is a key upgrade in our deployment strategy, offering substantial benefits:

1. **Scalability and Speed:** By storing large model files in the image layer, we eliminate the need for re-downloading upon each container startup. This efficiency reduces startup and shutdown times, ensuring our service is both responsive and cost-effective, perfect for Function as a Service (FaaS) scenarios. Further details on image caching are available in Modal's [Image caching and rebuilds guide](https://modal.com/docs/guide/custom-container#image-caching-and-rebuilds).

2. **Efficiency:** Model caching cuts down the time and resources used to fetch and load models, which is crucial in environments requiring rapid scaling.

### Implementing Model Caching

Here’s how we utilize Modal’s image caching to expedite deployment and service scaling:

```python
def download_model(model_id: str):
    import subprocess

    subprocess.run(
        [
            TABBY_BIN,
            "download",
            "--model",
            model_id,
        ]
    )


image = (
    Image.from_registry(
        IMAGE_NAME,
        add_python="3.11",
    )
    .env(TABBY_ENV)
    .dockerfile_commands("ENTRYPOINT []")
    .run_function(download_model, kwargs={"model_id": EMBEDDING_MODEL_ID})
    .run_function(download_model, kwargs={"model_id": CHAT_MODEL_ID})
    .run_function(download_model, kwargs={"model_id": MODEL_ID})
    .pip_install("asgi-proxy-lib")
)

app = App("tabby-server", image=image)
```

Modal determines the necessity to rebuild an image based on its definition changes. If unchanged, Modal pulls the previous version from cache, optimizing deployment processes.

## The Role of Persistent Volumes

Persistent volumes tackle several challenges inherent in FaaS environments:

1. **Data Persistence:** Frequent container startups and shutdowns typically disrupt user data and configuration continuity. Persistent volumes maintain this data intact across sessions. For more insights, check Modal’s [guide on persisting volumes](https://modal.com/docs/guide/volumes#persisting-volumes).

2. **User Experience:** Synchronized configuration files and essential data through persistent volumes eliminate the need for users to repeatedly configure settings, enhancing user experience and reliability.

3. **Operational Stability:** Providing a stable storage solution, persistent volumes are crucial for maintaining service reliability amidst frequent container cycling.

### Implementing Persistent Volumes

Here's our approach to utilizing Modal's persistent volumes to ensure data consistency and independence from the container lifecycle:

```python
data_volume = Volume.from_name("tabby-data", create_if_missing=True)
data_dir = "/data"

@app.function(
    gpu=GPU_CONFIG,
    allow_concurrent_inputs=10,
    container_idle_timeout=120,
    timeout=360,
    volumes={data_dir: data_volume},
    _allow_background_volume_commits=True,
    concurrency_limit=1,
)
```

The `create_if_missing=True` parameter lazily creates a volume if it doesn’t exist, while `_allow_background_volume_commits` allows for automatic data snapshotting and committing at regular intervals and upon container shutdown.

## The Complete App.py

The `app.py` script centralizes all configurations, model management, and service functionalities, making it a core component of our Modal deployment, you can see it in [Tabby GitHub repository](https://github.com/TabbyML/tabby/blob/main/website/blog/2024-07-10-enhanced-tabby-deployment-on-modal/app.py).

## Conclusion

These strategic enhancements on Modal not only optimize operational aspects but also significantly boost user experience by providing quicker startup times and reliable data persistence. By integrating model caching and persistent volumes, Tabby remains a sturdy and efficient tool in the ever-evolving serverless landscape.

For a deeper dive into these strategies, we recommend our [detailed tutorial](https://github.com/TabbyML/tabby/blob/main/website/docs/quick-start/installation/modal/index.md), which outlines a comprehensive guide to setting up your Tabby instance with these advanced features.

We hope this update inspires you to enhance your deployments similarly. Stay tuned for more updates, and happy coding with Tabby!
