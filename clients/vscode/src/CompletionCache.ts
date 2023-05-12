import { LinkedList } from "linked-list-typescript";
import { CompletionResponse, Choice } from "./generated";

type Range = {
  start: number;
  end: number;
};

export type CompletionCacheEntry = {
  documentId: any;
  promptRange: Range;
  prompt: string;
  completion: CompletionResponse;
};

export class CompletionCache {
  public static capacity = 10;
  private cache = new LinkedList<CompletionCacheEntry>();

  constructor() {}

  private refresh(entry: CompletionCacheEntry) {
    this.cache.remove(entry);
    this.cache.prepend(entry);
  }

  public add(entry: CompletionCacheEntry) {
    this.cache.prepend(entry);

    while (this.cache.length > CompletionCache.capacity) {
      this.cache.removeTail();
    }

  }

  public findCompatible(documentId: any, text: string, cursor: number): CompletionResponse | null {
    let hit: { entry: CompletionCacheEntry; compatibleChoices: Choice[] } | null = null;
    for (const entry of this.cache) {
      if (entry.documentId !== documentId) {
        continue;
      }
      // Check if text in prompt range has not changed
      if (text.slice(entry.promptRange.start, entry.promptRange.end) !== entry.prompt) {
        continue;
      }
      // Filter choices that start with inputed text after prompt
      const compatibleChoices = entry.completion.choices
        .filter((choice) => choice.text.startsWith(text.slice(entry.promptRange.end, cursor)))
        .map((choice) => {
          return {
            index: choice.index,
            text: choice.text.substring(cursor - entry.promptRange.end),
          };
        });
      if (compatibleChoices.length > 0) {
        hit = {
          entry,
          compatibleChoices,
        };
        break;
      }
    }
    if (hit) {
      this.refresh(hit.entry);
      return {
        id: hit.entry.completion.id,
        created: hit.entry.completion.created,
        choices: hit.compatibleChoices,
      };
    }
    return null;
  }
}
