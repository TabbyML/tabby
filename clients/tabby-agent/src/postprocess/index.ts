import { CompletionRequest, CompletionResponse } from "../Agent";
import { applyFilter } from "./filter";
import { removeDuplicatedContent } from "./removeDuplicatedContent";
import { dropBlank } from "./dropBlank";

export async function postprocess(
  request: CompletionRequest,
  response: CompletionResponse
): Promise<CompletionResponse> {
  return new Promise((resolve) => resolve(response))
    .then(applyFilter(removeDuplicatedContent(request)))
    .then(applyFilter(dropBlank()));
}
