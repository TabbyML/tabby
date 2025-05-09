import { PostprocessFilter, logger } from "./base";
import { CompletionResultItem, emptyCompletionResultItem } from "../solution";
import { isBlank } from "../../utils/string";

const repetitionTests = [
  /(.{3,}?)\1{5,}$/g, // match a 3+ characters pattern repeating 5+ times
  /(.{10,}?)\1{3,}$/g, // match a 10+ characters pattern repeating 3+ times
];

export function removeLineEndsWithRepetition(): PostprocessFilter {
  return (item: CompletionResultItem): CompletionResultItem => {
    // only test last non-blank line
    const inputLines = item.lines;
    let index = inputLines.length - 1;
    while (index >= 0 && isBlank(inputLines[index]!)) {
      index--;
    }
    if (index < 0) {
      return item;
    }
    // if matches repetition test, remove this line
    for (const test of repetitionTests) {
      const match = inputLines[index]!.match(test);
      if (match) {
        logger.trace("Remove line ends with repetition.", {
          inputLines,
          lineNumber: index,
          match,
        });
        if (index < 1) {
          return emptyCompletionResultItem;
        }
        return item.withText(inputLines.slice(0, index).join("").trimEnd());
      }
    }
    // no repetition found
    return item;
  };
}
