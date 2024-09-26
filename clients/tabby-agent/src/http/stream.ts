import { Readable } from "readable-stream";
import { readableFromWeb } from "readable-from-web";
import { EventSourceParserStream, ParsedEvent } from "eventsource-parser/stream";
import type { components as TabbyApiComponents } from "tabby-openapi/compatible";
import { getLogger } from "../logger";

const logger = getLogger("Stream");

export function readChatStream(stream: ReadableStream, signal?: AbortSignal): Readable {
  const eventStream = stream.pipeThrough(new TextDecoderStream()).pipeThrough(new EventSourceParserStream());
  const readableStream = readableFromWeb(eventStream, { objectMode: true });
  return readableStream.map(
    (event: ParsedEvent): string | undefined => {
      try {
        if (event.type === "event") {
          const chunk = JSON.parse(event.data) as TabbyApiComponents["schemas"]["ChatCompletionChunk"];
          const text = chunk.choices[0]?.delta.content;
          if (typeof text === "string") {
            return text;
          }
        }
      } catch (error) {
        logger.error("Failed to parse chat stream chunk.", error);
        logger.trace("Parsing failed with event:", { event });
      }
      return undefined;
    },
    { signal },
  );
}
