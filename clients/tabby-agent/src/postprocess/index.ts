import { CompletionRequest, CompletionResponse } from "../Agent";
import { applyFilter } from "./filter";
import { limitScopeByIndentation } from "./limitScopeByIndentation";
import { removeOverlapping } from "./removeOverlapping";
import { dropBlank } from "./dropBlank";

export async function postprocess(
  request: CompletionRequest,
  response: CompletionResponse
): Promise<CompletionResponse> {
  return new Promise((resolve) => resolve(response))
    .then(applyFilter(limitScopeByIndentation(request)))
    .then(applyFilter(removeOverlapping(request)))
    .then(applyFilter(dropBlank()));
}
