import { useEffect } from 'react'
import {
  createCallbacksTransformer,
  createStreamDataTransformer,
  StreamingTextResponse,
  type AIStreamCallbacksAndOptions
} from 'ai'

const serverUrl = process.env.NEXT_PUBLIC_TABBY_SERVER_URL || ''

export function usePatchFetch() {
  useEffect(() => {
    const fetch = window.fetch

    window.fetch = async function (url, options) {
      if (url !== '/api/chat') {
        return fetch(url, options)
      }

      const res = await fetch(`${serverUrl}/v1beta/chat/completions`, {
        ...options,
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      })

      const stream = StreamAdapter(res, undefined)
      return new StreamingTextResponse(stream)
    }
  }, [])
}

const utf8Decoder = new TextDecoder('utf-8')

async function processLines(
  lines: string[],
  controller: ReadableStreamDefaultController<string>
) {
  for (const line of lines) {
    const { content } = JSON.parse(line)
    controller.enqueue(content)
  }
}

async function readAndProcessLines(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  controller: ReadableStreamDefaultController<string>
) {
  let segment = ''

  while (true) {
    const { value: chunk, done } = await reader.read()
    if (done) {
      break
    }

    segment += utf8Decoder.decode(chunk, { stream: true })

    const linesArray = segment.split(/\r\n|\n|\r/g)
    segment = linesArray.pop() || ''

    await processLines(linesArray, controller)
  }

  if (segment) {
    const linesArray = [segment]
    await processLines(linesArray, controller)
  }

  controller.close()
}

function createParser(res: Response) {
  const reader = res.body?.getReader()

  return new ReadableStream<string>({
    async start(controller): Promise<void> {
      if (!reader) {
        controller.close()
        return
      }

      await readAndProcessLines(reader, controller)
    }
  })
}

function StreamAdapter(
  reader: Response,
  callbacks?: AIStreamCallbacksAndOptions
): ReadableStream {
  return createParser(reader)
    .pipeThrough(createCallbacksTransformer(callbacks))
    .pipeThrough(
      createStreamDataTransformer(callbacks?.experimental_streamData)
    )
}
