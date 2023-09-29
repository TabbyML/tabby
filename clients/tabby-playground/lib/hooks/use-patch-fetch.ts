import { type Message } from 'ai/react'
import { CohereStream, StreamingTextResponse } from 'ai'
import { useEffect } from 'react'

const serverUrl = process.env.NEXT_PUBLIC_TABBY_SERVER_URL || "http://localhost:8080"

export function usePatchFetch() {
    useEffect(() => {
        const fetch = window.fetch;

        window.fetch = async function (url, options) {
            if (url !== "/api/chat") {
                return fetch(url, options)
            }

            const { messages } = JSON.parse(options!.body as string);
            const res = await fetch(`${serverUrl}/v1beta/generate_stream`, {
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
    }, [])
}


function messagesToPrompt(messages: Message[]) {
    const instruction = messages[messages.length - 1].content
    const prompt = `Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n${instruction}\n\n### Response:`
    return prompt;
}