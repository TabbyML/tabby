import { CompletionContext, CompletionResponse } from "../CompletionContext";
import { AgentConfig } from "../AgentConfig";
import { applyFilter, applyChoiceFilter } from "./base";
import { removeRepetitiveBlocks } from "./removeRepetitiveBlocks";
import { removeRepetitiveLines } from "./removeRepetitiveLines";
import { removeLineEndsWithRepetition } from "./removeLineEndsWithRepetition";
import { removeDuplicatedBlockClosingLine } from "./removeDuplicatedBlockClosingLine";
import { limitScope } from "./limitScope";
import { formatIndentation } from "./formatIndentation";
import { trimSpace } from "./trimSpace";
import { trimMultiLineInSingleLineMode } from "./trimMultiLineInSingleLineMode";
import { dropDuplicated } from "./dropDuplicated";
import { dropBlank } from "./dropBlank";
import { calculateReplaceRange } from "./calculateReplaceRange";

export async function preCacheProcess(
  context: CompletionContext,
  _: AgentConfig["postprocess"],
  response: CompletionResponse,
): Promise<CompletionResponse> {
  return Promise.resolve(response)
    .then(applyFilter(trimMultiLineInSingleLineMode(), context))
    .then(applyFilter(removeLineEndsWithRepetition(), context))
    .then(applyFilter(dropDuplicated(), context))
    .then(applyFilter(trimSpace(), context))
    .then(applyFilter(dropBlank(), context));
}

export async function postCacheProcess(
  context: CompletionContext,
  config: AgentConfig["postprocess"],
  response: CompletionResponse,
): Promise<CompletionResponse> {
  return Promise.resolve(response)
    .then(applyFilter(removeRepetitiveBlocks(), context))
    .then(applyFilter(removeRepetitiveLines(), context))
    .then(applyFilter(limitScope(config["limitScope"]), context))
    .then(applyFilter(removeDuplicatedBlockClosingLine(), context))
    .then(applyFilter(formatIndentation(), context))
    .then(applyFilter(dropDuplicated(), context))
    .then(applyFilter(trimSpace(), context))
    .then(applyFilter(dropBlank(), context))
    .then(applyChoiceFilter(calculateReplaceRange(config["calculateReplaceRange"]), context));
}
