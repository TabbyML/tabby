import { type Message } from 'ai'
import { CohereStream, StreamingTextResponse } from 'ai'

export const runtime = 'edge'

export async function POST(req: Request) {
  const json = await req.json()
  const { messages } = json

  const res = await fetch("http://127.0.0.1:8080/v1beta/generate_stream", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      prompt: messagesToPrompt(messages),
    })
  })

  const stream = CohereStream(res, undefined);
  return new StreamingTextResponse(stream)
}

function messagesToPrompt(messages: Message[]) {
  const instruction = messages[messages.length - 1].content
  const prompt = `Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n${instruction}\n\n### Response:`
  return prompt;
}
