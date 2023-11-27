import { CompletionContext, CompletionResponse } from "../Agent";
import { AgentConfig } from "../AgentConfig";
import { isBrowser } from "../env";
import { applyFilter } from "./base";
import { removeRepetitiveBlocks } from "./removeRepetitiveBlocks";
import { removeRepetitiveLines } from "./removeRepetitiveLines";
import { removeLineEndsWithRepetition } from "./removeLineEndsWithRepetition";
import { limitScope } from "./limitScope";
import { trimSpace } from "./trimSpace";
import { dropDuplicated } from "./dropDuplicated";
import { dropBlank } from "./dropBlank";
import { calculateReplaceRangeByBracketStack } from "./calculateReplaceRangeByBracketStack";
import { calculateReplaceRangeBySyntax, supportedLanguages } from "./calculateReplaceRangeBySyntax";

export async function preCacheProcess(
  context: CompletionContext,
  config: AgentConfig["postprocess"],
  response: CompletionResponse,
): Promise<CompletionResponse> {
  return Promise.resolve(response)
    .then(applyFilter(removeLineEndsWithRepetition(context), context))
    .then(applyFilter(dropDuplicated(context), context))
    .then(applyFilter(trimSpace(context), context))
    .then(applyFilter(dropBlank(), context));
}

export async function postCacheProcess(
  context: CompletionContext,
  config: AgentConfig["postprocess"],
  response: CompletionResponse,
): Promise<CompletionResponse> {
  return Promise.resolve(response)
    .then(applyFilter(removeRepetitiveBlocks(context), context))
    .then(applyFilter(removeRepetitiveLines(context), context))
    .then(applyFilter(limitScope(context, config["limitScope"]), context))
    .then(applyFilter(dropDuplicated(context), context))
    .then(applyFilter(trimSpace(context), context))
    .then(applyFilter(dropBlank(), context));
}

export async function calculateReplaceRange(
  context: CompletionContext,
  config: AgentConfig["postprocess"],
  response: CompletionResponse,
): Promise<CompletionResponse> {
  return isBrowser || // syntax parser is not supported in browser yet
    !config["calculateReplaceRange"].experimentalSyntax ||
    !supportedLanguages.includes(context.language)
    ? calculateReplaceRangeByBracketStack(response, context)
    : calculateReplaceRangeBySyntax(response, context);
}
