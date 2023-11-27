import { CompletionContext } from "../Agent";
import { AgentConfig } from "../AgentConfig";
import { isBrowser } from "../env";
import { PostprocessFilter } from "./base";
import { limitScopeByIndentation } from "./limitScopeByIndentation";
import { limitScopeBySyntax, supportedLanguages } from "./limitScopeBySyntax";

export function limitScope(
  context: CompletionContext,
  config: AgentConfig["postprocess"]["limitScope"],
): PostprocessFilter {
  return isBrowser
    ? (input) => {
        // syntax parser is not supported in browser yet
        return limitScopeByIndentation(context, config["indentation"])(input);
      }
    : (input) => {
        if (config.experimentalSyntax && supportedLanguages.includes(context.language)) {
          return limitScopeBySyntax(context)(input);
        } else {
          return limitScopeByIndentation(context, config["indentation"])(input);
        }
      };
}
