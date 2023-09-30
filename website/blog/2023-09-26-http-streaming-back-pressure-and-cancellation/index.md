---
authors: [wwayne, icycodes, gyxlucy]

tags: [tech design]
---

# HTTP Streaming Back-Pressure and Cancellation in Code Completion

![Introduction](./intro.png)

## What is back-pressure?

![Back-pressure](./back-pressure.png)

Let's think about **_Black Friday_** 🛍️. During the time, online shops (_LLM_) will keep sending a bulk of deliveries to the express companies (_Server_), but express companies (_Server_) are only able to send a limited number of deliveries to customers (_Client_) every day. Finally, express companies (_Server_) will get into trouble of warehouse overstock (_out of memory_).

In many LLM applications, responses from LLM are usually large, so the server is always under the pressure of receiving too much data. Thus, we need to figure out a way to consume the data as soon as possible before server gets out of memory.

## How to handle back-pressure?

The solution is **Using Stream** and **Consume As You Use**. By using stream, we can receive the data time to time with a smaller piece instead of handling a bunch of data in one time.

![Stream](./stream.png)

**Stream between LLM and server** can reduce the memory usage of the server, and the **stream between server and client** can improve the user experience by a faster rendering.

```js
// Demo code for the stream between server and LLM

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// 1. Send stream response from LLM side
app.get("/api/llm", async (req, res) => {
  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Connection": "keep-alive",
    "Cache-Control": "no-cache"
  });

  let counter = 0;
  for (let i = 0; i < 10; i++) {
    res.write(`data: ${counter++}\n\n`);
    await sleep(100);
  }

  req.on("close", () => {
    res.end();
  });
});

// 2. Receive stream in the server side
app.get("/api/server", async (req, res) => {
  const response = await fetch("LLM_HOST/api/llm");
  const reader = response.body.getReader();

  while (true) {
    // 3. Get data from the stream when it is available
    const { done, value } = await reader.read();
    if (done) break;
    const chunk = new TextDecoder("utf-8").decode(value);
    // Do something with the chunk
  }
});
```

But only using stream can not solve the back-pressure issue in the server side. If the stream between server and LLM is faster than the stream between server and client, the server would still save too much data in the memory till out-of-memory.

![Stream-flush](./stream-flush.png)

That’s why we need to make use of `ReadableStream` and its method `pull()` to
manually fetch data from the stream when needed. implementing lazy fetching.

```js
// Demo code for the lazy-fetching in the server side

app.get("/api/server", async (req, res) => {
  // 1. Read stream from the LLM
  const response = await fetch("LLM_HOST/api/llm");
  const reader = response.body.getReader();

  // 2. Create a ReadableStream
  const serverStream = new ReadableStream({
    async pull(controller) {
      const { value, done } = await reader.read();
      if (done) {
        controller.close();
      } else {
        controller.enqueue(value);
      }
    }
  });

  const serverReader = serverStream.getReader();
  // 3. Assume we do fethcing every 1 second and only do 3 times
  for (let i = 0; i < 3; i++) {
    // 4. Read data from the ReadableStream, triggering the pull()
    const { value } = await serverReader.read();
    const chunk = new TextDecoder("utf-8").decode(value);
    // Do something with the chunk
    await sleep(1000);
  }
});
```

## What is cancellation?

Now we know the backend is made up by the stream, and the stream usually is expensive and time-consuming. What if the client abort the in-flight request because of the network issue or other intended behaviors? That’s why we need to implement the cancellation to stop the stream on time in order to save the computer resource.

![Cancellation](./cancellation.png)

## How to handle canellation?

The core idea is straightforward: on the server side, we need to listen to the `close` event and check if the connection is still valid before pulling data from the LLM stream.

```js
// Demo code in the server side

app.get("/api/server", async (req, res) => {
  let isConnectionClosed = false;

  // 1. Listen to the close event
  req.on("close", () => {
    isConnectionClosed = true;
  });

  const response = await fetch("LLM_HOST/api/llm");
  const reader = response.body.getReader();

  const serverStream = new ReadableStream({
    async pull(controller) {
      const { value, done } = await reader.read();
      if (done) {
        controller.close();
      } else {
        controller.enqueue(value);
      }
    }
  });

  const serverReader = serverStream.getReader();
  for (let i = 0; i < 3; i++) {
    // 2. Check if the connection closed before pulling the data
    if (isConnectionClosed) {
      // Here we can notify LLM stream to stop producing the data
      break;
    }
    const { value } = await serverReader.read();
    const chunk = new TextDecoder("utf-8").decode(value);
    // Do something with the chunk
    await sleep(1000);
  }

  res.json({});
});
```

## Implement back-pressure and cancellation for Tabby

In Tabby, we need to handle both back-pressure and cancellation for code completion so that we can react to users' inputs as soon as possible, while protecting the usage of the model to keep the performance.

On the client side, everytime we receive a new input from a user, we need to abort the previous query to the server and fetch a new response from LLM.

```js
// Demo code in the client side

let controller;

const callServer = (prompt) => {
  controller = new AbortController();
  const signal = controller.signal;
  // 2. calling server API to get the result with the prompt
  const response = await fetch("SERVER_HOST/v1/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ prompt })
    signal
  });
}

const onChange = (e) => {
  if (controller) controller.abort(); // Abort the previous request
  callServer(e.target.value);
};

// 1. Debounce the input 100ms for example
<input onChange={debounce(onChange)} />

```

On the server side, we need to listen to the `close` event and tell the LLM to stop generating results.

```js
// Demo code in the server side

app.post("/v1/completions", async (req, res) => {
  const { prompt } = req.body;

  // 1. Listen to the connection close event
  let isConnectionClosed = false;
  req.on("close", () => {
    isConnectionClosed = true;
  })

  // 2. Get response from LLM
  //	- cancel is a callable function to cancel LLM inference
  //	- text is a stream response from the LLM
  const { cancel, text } = llmInference(prompt);
  const reader = text.getReader();

  // 3. Create a ReadableStream
  const serverStream = new ReadableStream({
    async pull(controller) {
      const { value, done } = await reader.read();
      if (done) {
        controller.close();
      } else {
        controller.enqueue(value);
      }
    }
  });

  const serverReader = serverStream.getReader();
  for (true) {
    if (isConnectionClosed) {
      // 4. Notify LLM to stop the generating
      cancel();
      break;
    }
    const { value, done } = await serverReader();
    if (done) break;
    const chunk = new TextDecoder("utf-8").decode(value);
    // Do something with the chunk, e.g: send to the client
    await sleep(1000);
  }
})

```

## That's it

We would love to invite to join our Slack community! Please feel free to reach out to us on [Slack](https://join.slack.com/t/tabbycommunity/shared_invite/zt-1xeiddizp-bciR2RtFTaJ37RBxr8VxpA) - we have channels for discussing all aspects of the product and tech, and everyone is welcome to join the conversation.

Happy hacking 😁💪🏻
