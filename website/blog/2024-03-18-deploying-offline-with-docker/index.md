---
authors: [wwayne]
tags: [deployment]
---

# Deploying offline with Docker

<div align="center">

![No internet access](./no-internet.png)

</div>

Are you facing a situation where your work environment lacks internet access, and you're wondering if you can still deploy Tabby? Fear not, because the answer is YES! 🐱📣

## Prerequisite📋

* Docker installed on both the internet-connected computer and the offline computer.

## The Game Plan🚀

* Create a Docker image on a computer with internet access. 
* Transfer the image to your offline computer.
* Run the Docker image and let Tabby work its magic! ✨

## Step-by-Step Guide🐾

1. Create a new **Dockerfile** on a computer with internet access.

```docker
FROM tabbyml/tabby

ENV TABBY_MODEL_CACHE_ROOT=/models

RUN /opt/tabby/bin/tabby-cpu download --model StarCoder-1B
```

The **TABBY_MODEL_CACHE_ROOT** env var sets the directory for saving downloaded models. By setting `ENV TABBY_MODEL_CACHE_ROOT=/models`, we instruct Tabby to save the downloaded model files in the `/models` directory within the Docker container during the build process.

2. Build the Docker image which containing the model

```bash
docker build -t tabby-offline .
```

3. Save the Docker image to a tar file:

```bash
docker build -t tabby-offline .
```

4. Copy the `tabby-offline.tar` file to the computer without internet access.

5. Load the Docker image from the tar file:

```bash
docker load -i tabby-offline.tar
```

6. Run the Tabby container

```bash
docker run -it \
  --gpus all -p 8080:8080 -v $HOME/.tabby:/data \
  tabby-offline \
  serve --model StarCoder-1B --device cuda
```

If you encounter any further issues or have questions, feel free to explore our [community](https://slack.tabbyml.com/). Our friendly Tabby enthusiasts are always ready to lend a helping paw and guide you to the answers you seek! 😸💡
