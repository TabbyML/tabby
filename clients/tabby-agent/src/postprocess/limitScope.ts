import { CompletionContext } from "../CompletionContext";
import { AgentConfig } from "../AgentConfig";
import { isBrowser } from "../env";
import { PostprocessFilter } from "./base";
import { limitScopeByIndentation } from "./limitScopeByIndentation";
import { limitScopeBySyntax, supportedLanguages } from "./limitScopeBySyntax";

export function limitScope(config: AgentConfig["postprocess"]["limitScope"]): PostprocessFilter {
  return isBrowser
    ? (input: string, context: CompletionContext) => {
        // syntax parser is not supported in browser yet
        return limitScopeByIndentation(config["indentation"])(input, context);
      }
    : (input: string, context: CompletionContext) => {
        if (config.experimentalSyntax && supportedLanguages.includes(context.language)) {
          return limitScopeBySyntax()(input, context);
        } else {
          return limitScopeByIndentation(config["indentation"])(input, context);
        }
      };
}
