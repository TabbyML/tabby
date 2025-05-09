import { PostprocessFilter } from "./base";
import { CompletionResultItem } from "../solution";
import { CompletionContext } from "../contexts";
import { isBlank } from "../../utils/string";

export function trimSpace(): PostprocessFilter {
  return (item: CompletionResultItem, context: CompletionContext): CompletionResultItem => {
    const { currentLinePrefix, currentLineSuffix } = context;
    let trimmedInput = item.text;

    if (!isBlank(currentLinePrefix) && currentLinePrefix.match(/\s$/)) {
      trimmedInput = trimmedInput.trimStart();
    }
    if (isBlank(currentLineSuffix) || (!isBlank(currentLineSuffix) && currentLineSuffix.match(/^\s/))) {
      trimmedInput = trimmedInput.trimEnd();
    }
    if (trimmedInput !== item.text) {
      return item.withText(trimmedInput);
    }
    return item;
  };
}
