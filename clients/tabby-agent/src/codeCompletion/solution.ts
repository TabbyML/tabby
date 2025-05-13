import { splitLines, isBlank } from "../utils/string";
import type { components as TabbyApiComponents } from "tabby-openapi/compatible";
import type { CompletionContext, CompletionExtraContexts } from "./contexts";
import { CompletionItem, CompletionList, InlineCompletionItem, InlineCompletionList } from "../protocol";
import { CompletionItemKind } from "vscode-languageserver-protocol";

export class CompletionResultItem {
  // redundant quick access for text
  readonly lines: string[];
  readonly currentLine: string;

  constructor(
    readonly text: string,
    readonly eventId?: {
      completionId: string;
      choiceIndex: number;
    },
  ) {
    this.lines = splitLines(this.text);
    this.currentLine = this.lines[0] ?? "";
  }

  /**
   * Create a new CompletionResultItem from this item with the given text.
   * This method preserves the `eventId` property from the original item.
   * No other properties of the original item are carried over by design.
   */
  withText(text: string): CompletionResultItem {
    return new CompletionResultItem(text, this.eventId);
  }

  toCompletionItem(context: CompletionContext): CompletionItem | undefined {
    if (isBlank(this.text)) {
      return undefined;
    }

    const document = context.document;
    const position = context.position;
    const linePrefix = document.getText({
      start: { line: position.line, character: 0 },
      end: position,
    });
    const wordPrefix = linePrefix.match(/(\w+)$/)?.[0] ?? "";
    const insertText = context.selectedCompletionInsertion + this.text;

    const insertLines = splitLines(insertText);
    const firstLine = insertLines[0] || "";
    const secondLine = insertLines[1] || "";
    return {
      label: wordPrefix + firstLine,
      labelDetails: {
        detail: secondLine,
        description: "Tabby",
      },
      kind: CompletionItemKind.Text,
      documentation: {
        kind: "markdown",
        value: `\`\`\`\n${linePrefix + insertText}\n\`\`\`\n ---\nSuggested by Tabby.`,
      },
      textEdit: {
        newText: wordPrefix + insertText,
        range: {
          start: { line: position.line, character: position.character - wordPrefix.length },
          end: { line: position.line, character: position.character + context.lineEndReplaceLength },
        },
      },
      data: { eventId: this.eventId },
    };
  }

  toInlineCompletionItem(context: CompletionContext): InlineCompletionItem | undefined {
    if (isBlank(this.text)) {
      return undefined;
    }

    const position = context.position;
    const insertText = context.selectedCompletionInsertion + this.text;
    return {
      insertText,
      range: {
        start: position,
        end: { line: position.line, character: position.character + context.lineEndReplaceLength },
      },
      data: { eventId: this.eventId },
    };
  }
}

export class CompletionSolution {
  extraContext: CompletionExtraContexts = {};
  isCompleted: boolean = false;
  items: CompletionResultItem[] = [];

  toCompletionList(context: CompletionContext): CompletionList {
    return {
      isIncomplete: !this.isCompleted,
      items: this.items
        .map((item) => item.toCompletionItem(context))
        .filter((item): item is CompletionItem => item !== undefined),
    };
  }

  toInlineCompletionList(context: CompletionContext): InlineCompletionList {
    return {
      isIncomplete: !this.isCompleted,
      items: this.items
        .map((item) => item.toInlineCompletionItem(context))
        .filter((item): item is InlineCompletionItem => item !== undefined),
    };
  }
}

export const emptyCompletionResultItem = new CompletionResultItem("");

export function createCompletionResultItemFromResponse(
  response: TabbyApiComponents["schemas"]["CompletionResponse"],
): CompletionResultItem {
  const index = 0; // api always returns 0 or 1 choice
  return new CompletionResultItem(response.choices[index]?.text ?? "", {
    completionId: response.id,
    choiceIndex: response.choices[index]?.index ?? index,
  });
}
