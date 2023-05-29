import { LRUCache } from "lru-cache";
import hashObject from "object-hash";
import sizeOfObject from "object-sizeof";
import { CompletionRequest, CompletionResponse } from "./generated";
import { splitLines, splitWords } from "./utils";

type CompletionCacheKey = CompletionRequest;
type CompletionCacheValue = CompletionResponse;

export class CompletionCache {
  private cache: LRUCache<string, CompletionCacheValue>;
  private options = {
    maxSize: 1 * 1024 * 1024, // 1MB
    partiallyAcceptedCacheGeneration: {
      enabled: true,
      perCharacter: {
        lines: 1,
        words: 10,
        max: 30,
      },
      perWord: {
        lines: 1,
        max: 20,
      },
      perLine: {
        max: 3,
      },
    },
  };

  constructor() {
    this.cache = new LRUCache<string, CompletionCacheValue>({
      maxSize: this.options.maxSize,
      sizeCalculation: sizeOfObject,
    });
  }

  has(key: CompletionCacheKey): boolean {
    return this.cache.has(this.hash(key));
  }

  set(key: CompletionCacheKey, value: CompletionCacheValue): void {
    for (const entry of this.createCacheEntries(key, value)) {
      this.cache.set(this.hash(entry.key), entry.value);
    }
  }

  get(key: CompletionCacheKey): CompletionCacheValue | undefined {
    return this.cache.get(this.hash(key));
  }

  private hash(key: CompletionCacheKey): string {
    return hashObject(key);
  }

  private createCacheEntries(
    key: CompletionCacheKey,
    value: CompletionCacheValue
  ): { key: CompletionCacheKey; value: CompletionCacheValue }[] {
    const list = [{ key, value }];
    if (this.options.partiallyAcceptedCacheGeneration.enabled) {
      const entries = value.choices
        .map((choice) => {
          return this.calculatePartiallyAcceptedPositions(choice.text).map((position) => {
            return {
              prefix: choice.text.slice(0, position),
              suffix: choice.text.slice(position),
              choiceIndex: choice.index,
            };
          });
        })
        .flat()
        .reduce((grouped: { [key: string]: { suffix: string; choiceIndex: number }[] }, entry) => {
          grouped[entry.prefix] = grouped[entry.prefix] || [];
          grouped[entry.prefix].push({ suffix: entry.suffix, choiceIndex: entry.choiceIndex });
          return grouped;
        }, {});
      for (const prefix in entries) {
        const cacheKey = { ...key, prompt: key.prompt + prefix };
        const cacheValue = {
          ...value,
          choices: entries[prefix].map((choice) => {
            return {
              index: choice.choiceIndex,
              text: choice.suffix,
            };
          }),
        };
        list.push({
          key: cacheKey,
          value: cacheValue,
        });
      }
    }
    return list;
  }

  private calculatePartiallyAcceptedPositions(completion: string): number[] {
    const positions: number[] = [];
    const option = this.options.partiallyAcceptedCacheGeneration;

    const lines = splitLines(completion);
    let index = 0;
    let offset = 0;
    // `index < lines.length - 1` to exclude the last line
    while (index < lines.length - 1 && index < option.perLine.max) {
      offset += lines[index].length;
      positions.push(offset);
      index++;
    }

    const words = lines.slice(0, option.perWord.lines).map(splitWords).flat();
    index = 0;
    offset = 0;
    while (index < words.length && index < option.perWord.max) {
      offset += words[index].length;
      positions.push(offset);
      index++;
    }

    const characters = lines
      .slice(0, option.perCharacter.lines)
      .map(splitWords)
      .flat()
      .slice(0, option.perCharacter.words)
      .join("");
    offset = 1;
    while (offset < characters.length && offset < option.perCharacter.max) {
      positions.push(offset);
      offset++;
    }

    // distinct and sort ascending
    return positions.filter((v, i, arr) => arr.indexOf(v) === i).sort((a, b) => a - b);
  }
}
