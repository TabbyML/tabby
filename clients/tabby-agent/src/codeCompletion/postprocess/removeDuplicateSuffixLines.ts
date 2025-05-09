import { PostprocessFilter } from "./base";
import { CompletionResultItem } from "../solution";
import { CompletionContext } from "../contexts";
import { isBlank } from "../../utils/string";

export function removeDuplicateSuffixLines(): PostprocessFilter {
  return (item: CompletionResultItem, context: CompletionContext): CompletionResultItem => {
    const text = item?.text;
    const suffix = context?.suffix;

    if (text == null || suffix == null) {
      return item;
    }

    const originalLines = text.split("\n").map((line) => line || "");

    const suffixLines = (suffix || "")
      .split("\n")
      .map((line) => (line || "").trim())
      .filter((line) => !isBlank(line));

    if (suffixLines.length === 0) {
      return item;
    }

    const firstSuffixLine = suffixLines[0] || "";

    // iterate through lines from end to find potential match
    for (let i = originalLines.length - 1; i >= 0; i--) {
      const currentLine = originalLines[i] || "";
      if (!isBlank(currentLine) && currentLine === firstSuffixLine) {
        // check if subsequent lines also match with suffix
        let isFullMatch = true;
        for (let j = 0; j < suffixLines.length && i + j < originalLines.length; j++) {
          const suffixLine = suffixLines[j] || "";
          const textLine = originalLines[i + j] || "";
          if (suffixLine !== textLine) {
            isFullMatch = false;
            break;
          }
        }
        if (isFullMatch) {
          const remainingLines = originalLines.slice(0, i);
          return item.withText(remainingLines.join("\n"));
        }
      }
    }

    return item;
  };
}
