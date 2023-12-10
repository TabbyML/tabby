import { CompletionContext } from "../CompletionContext";
import { PostprocessFilter } from "./base";
import { isBlank } from "../utils";

export function trimSpace(): PostprocessFilter {
  return (input: string, context: CompletionContext) => {
    const { currentLinePrefix, currentLineSuffix } = context;
    let trimmedInput = input;
    if (!isBlank(currentLinePrefix) && currentLinePrefix.match(/\s$/)) {
      trimmedInput = trimmedInput.trimStart();
    }

    if (isBlank(currentLineSuffix) || (!isBlank(currentLineSuffix) && currentLineSuffix.match(/^\s/))) {
      trimmedInput = trimmedInput.trimEnd();
    }
    return trimmedInput;
  };
}
