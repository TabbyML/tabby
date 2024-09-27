import { splitLines, isBlank } from "../utils/string";
import type { components as TabbyApiComponents } from "tabby-openapi/compatible";
import type { CompletionContext } from "./contexts";

export type InlineCompletionItem = {
  insertText: string;
  // Range of the text to be replaced when applying the completion.
  // The range should be limited to the current line.
  range: {
    start: number;
    end: number;
  };
  data?: {
    eventId?: {
      completionId: string;
      choiceIndex: number;
    };
  };
};

export type InlineCompletionList = {
  // If the request is automatic, we will only provide one item, in this case,
  // `isIncomplete` will be `true`.
  // For the same completion context, if the request is manual, we will try
  // to provide multiple items, and `isIncomplete` will be `false`, whether the
  // item list actually contains multiple items, or not.
  isIncomplete: boolean;
  items: InlineCompletionItem[];
};

export class CompletionItem {
  // Shortcuts
  readonly text: string; // `replacePrefix` trimmed from `fullText`.
  readonly lines: string[]; // splitted lines of `text`.
  readonly currentLine: string; // first item of `lines`
  readonly isBlank: boolean; // whether the item is a blank line.

  constructor(
    // The context which the completion was generated for.
    readonly context: CompletionContext,
    // The full text of the completion.
    readonly fullText: string,
    // Prefix to replace, `text` must exactly start with `replacePrefix`.
    readonly replacePrefix: string = "",
    // Suffix to replace, in most case, `text` contains a subsequence of `replaceSuffix`.
    readonly replaceSuffix: string = "",
    // Extra event data
    readonly eventId?: {
      completionId: string;
      choiceIndex: number;
    },
  ) {
    this.text = fullText.substring(replacePrefix.length);
    this.lines = splitLines(this.text);
    this.currentLine = this.lines[0] ?? "";
    this.isBlank = isBlank(this.text);
  }

  static createBlankItem(context: CompletionContext): CompletionItem {
    return new CompletionItem(context, "");
  }

  static createFromResponse(
    context: CompletionContext,
    response: TabbyApiComponents["schemas"]["CompletionResponse"],
    index: number = 0,
  ): CompletionItem {
    return new CompletionItem(context, response.choices[index]?.text ?? "", "", "", {
      completionId: response.id,
      choiceIndex: response.choices[index]?.index ?? index,
    });
  }

  // Generate a CompletionItem based on this CompletionItem with modified `fullText`.
  withFullText(fullText: string): CompletionItem {
    return new CompletionItem(this.context, fullText, this.replacePrefix, this.replaceSuffix, this.eventId);
  }

  // Generate a CompletionItem based on this CompletionItem with modified `text`.
  withText(text: string): CompletionItem {
    return new CompletionItem(
      this.context,
      this.replacePrefix + text,
      this.replacePrefix,
      this.replaceSuffix,
      this.eventId,
    );
  }

  // Generate a CompletionItem based on this CompletionItem with modified `replaceSuffix`.
  withSuffix(replaceSuffix: string): CompletionItem {
    return new CompletionItem(this.context, this.fullText, this.replacePrefix, replaceSuffix, this.eventId);
  }

  // Generate a CompletionItem by trying to apply this CompletionItem to the new context.
  withContext(context: CompletionContext): CompletionItem {
    if (context.hash === this.context.hash) {
      return new CompletionItem(context, this.fullText, this.replacePrefix, this.replaceSuffix, this.eventId);
    } else {
      return CompletionItem.createBlankItem(context);
    }
  }

  // Generate a CompletionItem based on this CompletionItem.
  // Simulate as if the user typed over the same text as the completion.
  forward(chars: number): CompletionItem {
    if (chars <= 0) return this;
    const delta = this.text.substring(0, chars);
    // Forward in the current line
    if (chars < this.currentLine.length) {
      return new CompletionItem(
        this.context.forward(delta),
        this.fullText,
        this.replacePrefix + delta,
        this.replaceSuffix,
        this.eventId,
      );
    }
    // Forward to next lines
    const lastLineStart = delta.lastIndexOf("\n") + 1;
    const lastLine = delta.substring(lastLineStart);
    let whiteSpaces = lastLine.search(/\S/);
    if (whiteSpaces < 0) {
      whiteSpaces = lastLine.length;
    }
    const lastLineNonSpaceStart = lastLineStart + whiteSpaces;
    const fullText = this.text.substring(lastLineNonSpaceStart);
    const lastLineNonSpace = delta.substring(lastLineNonSpaceStart);
    return new CompletionItem(
      this.context.forward(delta),
      fullText,
      lastLineNonSpace,
      this.replaceSuffix,
      this.eventId,
    );
  }

  isSameWith(other: CompletionItem): boolean {
    return this.context.hash === other.context.hash && this.text === other.text;
  }

  toInlineCompletionItem(): InlineCompletionItem {
    return {
      insertText: this.fullText,
      range: {
        start: this.context.currentLinePrefix.endsWith(this.replacePrefix)
          ? this.context.position - this.replacePrefix.length
          : this.context.position,
        end: this.context.currentLineSuffix.startsWith(this.replaceSuffix)
          ? this.context.position + this.replaceSuffix.length
          : this.context.position,
      },
      data: { eventId: this.eventId },
    };
  }
}

export class CompletionSolution {
  items: CompletionItem[] = [];
  isCompleted: boolean = false;

  constructor(
    readonly context: CompletionContext,
    items: CompletionItem[] = [],
    isCompleted = false,
  ) {
    this.add(...items);
    this.isCompleted = isCompleted;
  }

  static merge(base: CompletionSolution, addition: CompletionSolution): CompletionSolution {
    if (base.context.hash === addition.context.hash) {
      return new CompletionSolution(
        base.context,
        [...base.items, ...addition.items],
        base.isCompleted || addition.isCompleted,
      );
    } else {
      return base;
    }
  }

  add(...items: CompletionItem[]): void {
    this.items.push(
      ...items
        .map((item) => item.withContext(this.context))
        .filter((item, index, arr) => {
          return (
            !item.isBlank &&
            // deduplicate
            arr.findIndex((i) => i.isSameWith(item)) === index &&
            this.items.findIndex((i) => i.isSameWith(item)) === -1
          );
        }),
    );
  }

  // Generate a CompletionSolution by trying to apply this CompletionSolution to the new context.
  withContext(context: CompletionContext): CompletionSolution {
    if (context.hash === this.context.hash) {
      return new CompletionSolution(context, this.items, this.isCompleted);
    } else {
      return new CompletionSolution(context);
    }
  }

  // Generate a CompletionSolution by replacing all items.
  withItems(...items: CompletionItem[]): CompletionSolution {
    return new CompletionSolution(this.context, items, this.isCompleted);
  }

  toInlineCompletionList(): InlineCompletionList {
    return {
      isIncomplete: !this.isCompleted,
      items: this.items.map((item) => item.toInlineCompletionItem()),
    };
  }
}
