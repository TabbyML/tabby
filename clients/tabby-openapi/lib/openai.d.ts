import type { OpenAI } from "openai";
type ChatCompletionRequest = OpenAI.ChatCompletionCreateParams;
type ChatCompletionChunk = OpenAI.ChatCompletionChunk;
type ChatCompletionRequestMessage = OpenAI.ChatCompletionMessageParam;
export { ChatCompletionRequest, ChatCompletionChunk, ChatCompletionRequestMessage };
