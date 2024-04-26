import { CompletionContext } from "../CompletionContext";
import { AgentConfig } from "../AgentConfig";
import { PostprocessFilter } from "./base";
import { limitScopeByIndentation } from "./limitScopeByIndentation";

export function limitScope(config: AgentConfig["postprocess"]["limitScope"]): PostprocessFilter {
  return async (input: string, context: CompletionContext) => {
    return limitScopeByIndentation(config)(input, context);
  };
}
