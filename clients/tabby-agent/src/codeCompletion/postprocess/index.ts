import type { ConfigData } from "../../config/type";
import type { CompletionResultItem } from "../solution";
import type { CompletionContext, CompletionExtraContexts } from "../contexts";
import type { PostprocessFilter, PostprocessFilterFactory } from "./base";
import "../../utils/array";
import { removeRepetitiveBlocks } from "./removeRepetitiveBlocks";
import { removeRepetitiveLines } from "./removeRepetitiveLines";
import { removeLineEndsWithRepetition } from "./removeLineEndsWithRepetition";
import { removeDuplicatedBlockClosingLine } from "./removeDuplicatedBlockClosingLine";
import { limitScope } from "./limitScope";
import { formatIndentation } from "./formatIndentation";
import { trimSpace } from "./trimSpace";
import { trimMultiLineInSingleLineMode } from "./trimMultiLineInSingleLineMode";
import { dropDuplicated } from "./dropDuplicated";
import { dropMinimum } from "./dropMinimum";
import { removeDuplicateSuffixLines } from "./removeDuplicateSuffixLines";
import { normalizeIndentation } from "./normalizeIndentation";

export interface ItemsWithContext {
  items: CompletionResultItem[];
  context: CompletionContext;
  extraContext: CompletionExtraContexts;
}
type ItemsFilter = (params: ItemsWithContext) => Promise<ItemsWithContext>;

function createListFilter(filterFactory: PostprocessFilterFactory, config: ConfigData["postprocess"]): ItemsFilter {
  const filter: PostprocessFilter = filterFactory(config);
  return async (params: ItemsWithContext): Promise<ItemsWithContext> => {
    const processed = await params.items.mapAsync(async (item) => {
      return await filter(item, params.context, params.extraContext);
    });
    return { items: processed, context: params.context, extraContext: params.extraContext };
  };
}

export async function preCacheProcess(
  items: CompletionResultItem[],
  context: CompletionContext,
  extraContext: CompletionExtraContexts,
  config: ConfigData["postprocess"],
): Promise<CompletionResultItem[]> {
  const applyFilter = (filterFactory: PostprocessFilterFactory): ItemsFilter => {
    return createListFilter(filterFactory, config);
  };
  const result = await Promise.resolve({ items, context, extraContext })
    .then(applyFilter(trimMultiLineInSingleLineMode))
    .then(applyFilter(removeLineEndsWithRepetition))
    .then(applyFilter(dropDuplicated))
    .then(applyFilter(trimSpace))
    .then(applyFilter(dropMinimum));
  return result.items;
}

export async function postCacheProcess(
  items: CompletionResultItem[],
  context: CompletionContext,
  extraContext: CompletionExtraContexts,
  config: ConfigData["postprocess"],
): Promise<CompletionResultItem[]> {
  const applyFilter = (filterFactory: PostprocessFilterFactory): ItemsFilter => {
    return createListFilter(filterFactory, config);
  };
  const result = await Promise.resolve({ items, context, extraContext })
    .then(applyFilter(removeRepetitiveBlocks))
    .then(applyFilter(removeRepetitiveLines))
    .then(applyFilter(limitScope))
    .then(applyFilter(removeDuplicatedBlockClosingLine))
    .then(applyFilter(formatIndentation))
    .then(applyFilter(normalizeIndentation))
    .then(applyFilter(dropDuplicated))
    .then(applyFilter(trimSpace))
    .then(applyFilter(removeDuplicateSuffixLines))
    .then(applyFilter(dropMinimum));
  return result.items;
}
