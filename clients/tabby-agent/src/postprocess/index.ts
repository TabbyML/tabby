import { CompletionRequest, CompletionResponse } from "../Agent";
import { buildContext, applyFilter } from "./base";
import { removeRepetitiveBlocks } from "./removeRepetitiveBlocks";
import { removeRepetitiveLines } from "./removeRepetitiveLines";
import { removeLineEndsWithRepetition } from "./removeLineEndsWithRepetition";
import { limitScopeByIndentation } from "./limitScopeByIndentation";
import { trimSpace } from "./trimSpace";
import { removeOverlapping } from "./removeOverlapping";
import { dropDuplicated } from "./dropDuplicated";
import { dropBlank } from "./dropBlank";

export async function preCacheProcess(
  request: CompletionRequest,
  response: CompletionResponse,
): Promise<CompletionResponse> {
  const context = buildContext(request);
  return Promise.resolve(response)
    .then(applyFilter(removeLineEndsWithRepetition(context)))
    .then(applyFilter(dropDuplicated(context)))
    .then(applyFilter(trimSpace(context)))
    .then(applyFilter(removeOverlapping(context)))
    .then(applyFilter(dropBlank()));
}

export async function postprocess(
  request: CompletionRequest,
  response: CompletionResponse,
): Promise<CompletionResponse> {
  const context = buildContext(request);
  return Promise.resolve(response)
    .then(applyFilter(removeRepetitiveBlocks(context)))
    .then(applyFilter(removeRepetitiveLines(context)))
    .then(applyFilter(limitScopeByIndentation(context)))
    .then(applyFilter(trimSpace(context)))
    .then(applyFilter(removeOverlapping(context)))
    .then(applyFilter(dropBlank()));
}
