import { CompletionContext } from "../CompletionContext";
import { PostprocessFilter } from "./base";

// remove the newline suffix if last char is \n
function trimEndingNewline(input: string): string {
  return input.endsWith("\n") ? input.slice(0, -1) : input;
}

function trimSimilarEndingChar(a: string, b: string): string {
  let i = a.length - 1;
  let j = b.length - 1;

  // Start from the end of both strings and move backwards
  while (i >= 0 && j >= 0) {
    if (a[i] === b[j]) {
      i--;
      j--;
    } else {
      break; // Stop when characters don't match
    }
  }

  // Trim the similar part, +1 because slice stops before the end index
  // and we decreased j one too many times in the last iteration of the loop
  return b.slice(0, j + 1);
}

export function removeDuplicatedLineSuffix(): PostprocessFilter {
  return (input: string, context: CompletionContext) => {
    // this filter is only for single line completions
    if (input.includes("\n")) return input;
    const currentLineSuffix = trimEndingNewline(context.currentLineSuffix);
    return trimSimilarEndingChar(currentLineSuffix, input);
  };
}
