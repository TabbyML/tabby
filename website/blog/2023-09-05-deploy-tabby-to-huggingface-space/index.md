---
slug: deploy-tabby-to-huggingface-space.md
title: Deploying a Tabby Instance in Hugging Face Spaces
authors:
  - name: Rand Xie
    title: Contributor of Tabby
    url: https://github.com/randxie
    image_url: https://github.com/randxie.png
  - meng
tags: [deployment]
---

[Hugging Face Spaces](https://huggingface.co/spaces) provides an easy way for anyone to host their machine learning models.
In this blog post, we are going to show you how to deploy a Tabby instance in Hugging Face Spaces. If you have not heard of Tabby, it’s an open-source Github Copilot alternative that supports code completion.
Check out more details [here](https://github.com/TabbyML/tabby).

### How it works

Let’s firstly take a look at what steps are needed to deploy a Tabby instance in Hugging Face. It’s super easy and you don’t need much coding knowledge. Buckle up and let’s get started.

**Step 1:** Create a Hugging Face account. You will need the account to create your Space. Sign up at [this link](https://huggingface.co/welcome) if you don’t have a Hugging Face account.

**Step 2:** Create a new Hugging Face Space ([link](https://huggingface.co/new-space)). Spaces are code repositories that host application code for Machine Learning demos.

**Step 3:** Create a Dockerfile to capture your machine learning models’ logic, and bring up a server to serve requests.

**Step 4:** After the Dockerfile is built, you will be able to send requests to the APIs.

That’s it! With the hosted APIs, you will be able to call these APIs in your apps. Next, we will deep dive into each step with screenshots!!
**Everything will be done in the Hugging Face UI. No local setup is needed.**

### Deep Dive

#### Create a new Space

After you create a Hugging Face account, you should be able to see the following page by clicking this [link](https://huggingface.co/new-space).
The owner name will be your account name. Fill in a Space name, e.g. "tabbyml", and select Docker as Space SDK. Then click "Create Space" at the bottom.

![Create a new Space](./new-space.png)

#### Uploading Dockerfile

For advanced users, you can leverage the Git workspace. In this blog post, we will show you the UI flow instead. After you click the "Create a Space" in the last step, you will be directed to this page. Just ignore the main text and click the "Files" on the top right corner.

![Docker Space](./docker-space.png)

After clicking on the "Files", you will be able to see a "Add file" button, click that, then click on "Create a new file"

![Empty Space](./empty-space.png)

Then you will be redirected to the page below. Set the filename to “Dockerfile” and copy the content to the “Edit” input box. You can copy the code from the [appendix](#dockerfile) here. Once ready, click the button “Commit new file to main” on the bottom.

![Edit Dockerfile](./edit-dockerfile.png)

#### Edit Readme

You also need to add a new line the README.md. Click the “edit” button in the README.md file.


![Empty README](./empty-readme.png)

Add this line “app_port: 8080” after “sdk: docker”

![Edit README](./edit-readme.png)

Click the button “Commit to main” to save the changes.

#### Verify Tabby is running

Click on the “App” button, you should be able to see the container is building:

![Space Building](./building.png)

If the App is up successfully, you should see this page:

![Tabby Swagger](./tabby-swagger.png)

#### Call the API to do Code Completion

Now, you are able to call the completion API. The full URL is https://{YOUR-ACCOUNT-NAME}-tabbyml.hf.space/v1/completions. In this post, the URL is https://randxie-tabbyml.hf.space/v1/completions.

To test if your APIs are up and running, use [this online tool](https://reqbin.com/curl) to send curl commands:

![curl](./curl.png)

The full curl command can be found in the [appendix](#curl-command) as well. Make sure you have changed the URL to match with your Hugging Face Spaces settings!!

### Conclusion
In this post, we covered the detailed steps for deploying a Tabby instance to Hugging Face Spaces. By following these steps, anyone is able to bring up their own code completion APIs easily.

### Appendix

#### Dockerfile
```Dockerfile
FROM tabbyml/tabby

RUN useradd -m -u 1000 user

USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

ENV TABBY_ROOT=$HOME/app

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

CMD ["serve", ”--device", "cuda", "--model", "TabbyML/SantaCoder-1B"]

```

#### CURL Command
```bash
curl -L 'https://randxie-tabbyml.hf.space/v1/completions' \
-H 'Content-Type: application/json' \
-H 'Accept: application/json' \
-d '{
  "language": "python",
  "segments": {
    "prefix": "def fib(n):\n    ",
    "suffix": "\n        return fib(n - 1) + fib(n - 2)"
  }
}'
```
