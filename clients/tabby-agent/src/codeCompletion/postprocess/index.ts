import type { CompletionItem } from "../solution";
import type { ConfigData } from "../../config/type";
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
import { calculateReplaceRange } from "./calculateReplaceRange";

type ItemListFilter = (items: CompletionItem[]) => Promise<CompletionItem[]>;

function createListFilter(filterFactory: PostprocessFilterFactory, config: ConfigData["postprocess"]): ItemListFilter {
  const filter: PostprocessFilter = filterFactory(config);
  return async (items: CompletionItem[]): Promise<CompletionItem[]> => {
    return await items.mapAsync(filter);
  };
}

export async function preCacheProcess(
  items: CompletionItem[],
  config: ConfigData["postprocess"],
): Promise<CompletionItem[]> {
  const applyFilter = (filterFactory: PostprocessFilterFactory): ItemListFilter => {
    return createListFilter(filterFactory, config);
  };
  return Promise.resolve(items)
    .then(applyFilter(trimMultiLineInSingleLineMode))
    .then(applyFilter(removeLineEndsWithRepetition))
    .then(applyFilter(dropDuplicated))
    .then(applyFilter(trimSpace))
    .then(applyFilter(dropMinimum));
}

export async function postCacheProcess(
  items: CompletionItem[],
  config: ConfigData["postprocess"],
): Promise<CompletionItem[]> {
  const applyFilter = (filterFactory: PostprocessFilterFactory): ItemListFilter => {
    return createListFilter(filterFactory, config);
  };
  return Promise.resolve(items)
    .then(applyFilter(removeRepetitiveBlocks))
    .then(applyFilter(removeRepetitiveLines))
    .then(applyFilter(limitScope))
    .then(applyFilter(removeDuplicatedBlockClosingLine))
    .then(applyFilter(formatIndentation))
    .then(applyFilter(dropDuplicated))
    .then(applyFilter(trimSpace))
    .then(applyFilter(dropMinimum))
    .then(applyFilter(calculateReplaceRange));
}
