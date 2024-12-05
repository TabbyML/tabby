import type { OpenAI } from "openai";

type ChatCompletionRequest = OpenAI.ChatCompletionCreateParams;

type ChatCompletionChunk = OpenAI.ChatCompletionChunk;

// Omit `name` and mark as optional.
// However, `name` is required when the `role` is `function`.
// This patch is for compatible with the type `Message` in https://www.npmjs.com/package/ai
type ChatCompletionRequestMessage = Omit<OpenAI.ChatCompletionMessageParam, "name"> & {
  name?: string;
};

export { ChatCompletionRequest, ChatCompletionChunk, ChatCompletionRequestMessage };
