---
authors: 
  - wwayne
  - icycodes
  - gyxlucy

tags: [tech design]
---
# HTTP Streaming Back-Pressure and Cancellation in Code Completion

![back-pressure](./intro.png)


## What is back-pressure?

![back-pressure](./back-pressure.png)


You can understand the concept better if you thinking about ***Black Friday***. During that time, the online shops (*LLM*) will keep sending a bulk of deliveries to the express companies (*Server*), but the express companies (*Server*) are only able to send a limited number of deliveries to customers (*Client*) every day. Finally,  express companies (*Server*) will get into trouble of warehouse overstock (*out of memory*).

Many applications produce a large amount of LLM responses so the server

In many LLM applications, responses from LLM are usually large, so the server is always under the pressure of receiving too much data and we need to figure out a way to consume the data as soon as possible before out of memory.


## How to handle back-pressure?

The solution is **Using Stream** and **Consume As You Use**. By using stream, we can receive the data time to time with a smaller piece instead of handling a bunch of data in one time.

![back-pressure](./stream.png)


Stream between LLM and server can reduce the memory usage of the server, and the stream between server and client can improve the user experience by a faster rendering.



```JavaScript
// Demo code for the stream between server and LLM

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Stream response from LLM
app.get('/api/llm', async (req, res) => {
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Connection': 'keep-alive',
    'Cache-Control': 'no-cache'
  });
  
  let counter = 0;
  for (let i = 0; i < 10; i++) {
    res.write(`data: ${counter++}\n\n`);
    await sleep(100);
  }

  req.on('close', () => {
    res.end();
  });
})

// Handle stream in server side
app.get('/api/server', async (req, res) => {
  const response = await fetch('http://localhost:8000/api/llm');
  const reader = response.body.getReader();

  while (true) {
		// Get data from stream when it available
    const {done, value} = await reader.read();
    if (done) {
      break;
    }
    const chunk = new TextDecoder("utf-8").decode(value);
    console.log(chunk); 
  }
})

```


But only using stream can **not** solve the back-pressure issue in the server side, because if the stream between server and LLM is faster than the stream between server and client, the server would still saving too much data in the memory till out-of-memory.

![stream-flush](./stream-flush.png)


That’s why we need to make use of `ReadableStream` and its method `pull` to
manually fetch the data from the stream when needed.


```JavaScript

// Demo code for the server
app.get('/api/server', async (req, res) => {
	// 1. Read stream from LLM
  const response = await fetch('http://localhost:8000/api/llm');
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

	// 3. Custom fetching-data-from-llm logic based on the frontend needs
  for (let i = 0; i < 3; i++) {
		// Trigger the `pull` function and fetch the data from stream 
    const { value } = await serverReader.read();
    const chunk = new TextDecoder("utf-8").decode(value);
    console.log(`read ${chunk}`); 
 
    await sleep(1000);
  }
})

```


## What is cancellation?

Now we know the backend is made up by the stream, and the stream usually is expensive and time-consuming. So what if the client abort the in-flight request because of the network issue or other intended behaviors? That’s why we need to implement the cancellation to stop the stream on time to save the computer resource.

![cancellation](./cancellation.png)

## How to handle canellation?

The core idea is straightforward: on the server side, we need to listen to the `close` event and check if the connection is still valid before pulling data from the LLM stream.

```JavaScript
app.get('/api/server', async (req, res) => {
  let isConnectionClosed = false

	// 1. Listen to the close event
  req.on('close', () => {
    console.log('close event')
    isConnectionClosed = true
  })

  const response = await fetch('http://localhost:8000/api/llm');
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
      // Notify LLM stream to stop producing the data
			break;
		}
    const { value } = await serverReader.read();
    const chunk = new TextDecoder("utf-8").decode(value);
    console.log(`read ${chunk}`); 
 
    await sleep(1000);
  }

  res.json({})
})
```


## Implement back-pressure and cancellation for Tabby

In Tabby, we need to handle both back- pressure and cancellation for the code auto- completion so that we can react to the user’s input as soon as possible, while protecting the usage of the model to keep the performance.

In the client side, everytime we got a new input from the user, we need to abort the previous query to the server and fetching a new response from the LLM.

```JavaScript
let controller;

const callServer = (prompt) => {
  controller = new AbortController();
  const signal = controller.signal;
	// 3. calling server API to get the result with the prompt
  const response = await fetch('/v1/completions', {
		method: "POST",
		headers: {
	    'Content-Type': 'application/json'
	  },
	  body: JSON.stringify({ prompt })
    signal
  });
}

const onChange = (e) => {
	// 2. Abort the previous request
  if (controller) controller.abort();
  callServer(e.target.value)
}

// 1. Debounce the input 100ms for example
<input onChange={debounce(onChange)} />

```


In the server side, we need to listen to the `close` event and tell the LLM to stop
generating the result.

```JavaScript
app.post('/v1/completions', async (req, res) => {
	const { prompt } = req.body

	// 1. Listen to the connection close event
  let isConnectionClosed = false
  req.on('close', () => {
    isConnectionClosed = true
  })

	// 2. Get the methods from LLM
	const { cancel, text } = llmInference(prompt)

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
			// 5. Notify LLM to stop the generating
			cancel()
			break;
		}
    const { value, done } = await text();
		if (done) break;
    const chunk = new TextDecoder("utf-8").decode(value);
    // 4. send chunk to the frontend as stream data
 
    await sleep(1000);
  }
})

```


That’s it :D

<center>

<img src="done.png" width="300" height="250">
</center>