import { AgentConfig } from "../AgentConfig";
import { PostprocessChoiceFilter } from "./base";
import { calculateReplaceRangeByBracketStack } from "./calculateReplaceRangeByBracketStack";

export function calculateReplaceRange(
  _config: AgentConfig["postprocess"]["calculateReplaceRange"],
): PostprocessChoiceFilter {
  return async (choice, context) => {
    return calculateReplaceRangeByBracketStack(choice, context);
  };
}
