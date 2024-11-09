import { PostprocessFilter } from "./base";
import { CompletionItem } from "../solution";
import { isBlank } from "../../utils/string";
import { getLogger } from "../../logger";

export function removeDuplicateSuffixLines(): PostprocessFilter {
  return (item: CompletionItem): CompletionItem => {
    const log = getLogger("removeDuplicateSuffixLines");
    log.info("Processing item" + JSON.stringify(item?.text || ""));

    const text = item?.text;
    const suffix = item?.context?.suffix;

    if (text == null || suffix == null) {
      return item;
    }

    const originalLines = text.split("\n").map((line) => line || "");
    const trimmedLines = originalLines.map((line) => (line || "").trim());

    const suffixLines = (suffix || "")
      .split("\n")
      .map((line) => (line || "").trim())
      .filter((line) => !isBlank(line));

    if (suffixLines.length === 0) {
      return item;
    }

    const firstSuffixLine = suffixLines[0] || "";

    // iterate through lines from end to find potential match
    for (let i = trimmedLines.length - 1; i >= 0; i--) {
      const currentLine = trimmedLines[i] || "";
      if (!isBlank(currentLine) && currentLine === firstSuffixLine) {
        // check if subsequent lines also match with suffix
        let isFullMatch = true;
        for (let j = 0; j < suffixLines.length && i + j < trimmedLines.length; j++) {
          const suffixLine = suffixLines[j] || "";
          const textLine = trimmedLines[i + j] || "";

          if (suffixLine !== textLine) {
            isFullMatch = false;
            break;
          }
        }

        // if all checked lines match, check for code structure
        if (isFullMatch) {
          const remainingLines = originalLines.slice(0, i);
          const lastLine = remainingLines[remainingLines.length - 1] || "";

          // skip empty last lines
          if (isBlank(lastLine.trim())) {
            return item;
          }

          // preserve code block structure
          if (lastLine.includes("{") || currentLine.includes("}")) {
            return item;
          }

          return item.withText(remainingLines.join("\n"));
        }
      }
    }

    return item;
  };
}
