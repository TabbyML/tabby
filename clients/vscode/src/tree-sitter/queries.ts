import { SupportedLanguage } from "./grammars";
import { goQueries } from "./queries/go";
import { javascriptQueries } from "./queries/javascript";
import { pythonQueries } from "./queries/python";

export type QueryName = "singlelineTriggers" | "intents" | "documentableNodes";

/**
 * Completion intents sorted by priority.
 * Top-most items are used if capture group ranges are identical.
 */
export const intentPriority = [
  "function.name",
  "function.parameters",
  "function.body",
  "type_declaration.name",
  "type_declaration.body",
  "arguments",
  "import.source",
  "comment",
  "pair.value",
  "argument",
  "parameter",
  "parameters",
  "jsx_attribute.value",
  "return_statement.value",
  "return_statement",
  "string",
] as const;

/**
 * Completion intent label derived from the AST nodes before the cursor.
 */
export type CompletionIntent = (typeof intentPriority)[number];

export const languages: Partial<
  Record<SupportedLanguage, Record<QueryName, string>>
> = {
  ...javascriptQueries,
  ...goQueries,
  ...pythonQueries,
} as const;
