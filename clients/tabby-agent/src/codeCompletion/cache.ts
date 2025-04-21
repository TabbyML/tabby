import type { TextDocuments } from "vscode-languageserver";
import type { TextDocument } from "vscode-languageserver-textdocument";
import { buildCompletionContextWithAppend, type CompletionContext } from "./contexts";
import { LRUCache } from "lru-cache";
import hashObject from "object-hash";
import { CompletionSolution, CompletionResultItem } from "./solution";

export class CompletionCache extends LRUCache<string, CompletionSolution> {
  constructor(options?: { max?: number; ttl?: number }) {
    const max = options?.max ?? 100;
    const ttl = options?.ttl ?? 5 * 60 * 1000; // 5 minutes
    super({
      max,
      ttl,
    });
  }
}

export function calculateCompletionContextHash(
  context: CompletionContext,
  textDocuments: TextDocuments<TextDocument>,
): string {
  return hashObject({
    document: {
      uri: context.document.uri,
      prefix: context.prefix,
      suffix: context.suffix,
    },
    otherDocuments: textDocuments
      .all()
      .filter((doc) => doc.uri !== context.document.uri)
      .map((doc) => ({
        uri: doc.uri,
        version: doc.version,
      })),
  });
}

export function generateForwardingContexts(
  context: CompletionContext,
  items: CompletionResultItem[],
  maxForwardingChars = 50,
): {
  context: CompletionContext;
  items: CompletionResultItem[];
}[] {
  const forwarding: { appending: string; remaining: string; eventId: CompletionResultItem["eventId"] }[] = [];
  for (const item of items) {
    // Forward at current line
    const steps = Math.min(maxForwardingChars, item.currentLine.length);
    for (let chars = 1; chars < steps; chars++) {
      forwarding.push({
        appending: item.currentLine.slice(0, chars),
        remaining: item.currentLine.slice(chars),
        eventId: item.eventId,
      });
    }
    if (item.lines.length > 2) {
      // current line end
      forwarding.push({
        appending: item.currentLine.slice(0, item.currentLine.length - 1),
        remaining: item.currentLine.slice(item.currentLine.length - 1),
        eventId: item.eventId,
      });
      // next line start
      forwarding.push({
        appending: item.currentLine.slice(0, item.currentLine.length),
        remaining: item.currentLine.slice(item.currentLine.length),
        eventId: item.eventId,
      });
      // next line start, after indent spaces
      const nextLine = item.lines[1]!;
      let spaces = nextLine.search(/\S/);
      if (spaces < 0) {
        spaces = nextLine.length - 1;
      }
      forwarding.push({
        appending: item.currentLine.slice(0, item.currentLine.length + spaces),
        remaining: item.currentLine.slice(item.currentLine.length + spaces),
        eventId: item.eventId,
      });
    }
  }

  const groupedForwarding = new Map<
    string,
    { appending: string; remaining: string; eventId: CompletionResultItem["eventId"] }[]
  >();
  for (const entry of forwarding) {
    if (!groupedForwarding.has(entry.appending)) {
      groupedForwarding.set(entry.appending, []);
    }
    groupedForwarding.get(entry.appending)?.push(entry);
  }

  return Array.from(groupedForwarding.entries()).map(([appending, entries]) => {
    const updatedContext = buildCompletionContextWithAppend(context, appending);
    const results = entries.map((entry) => {
      return new CompletionResultItem(entry.remaining, entry.eventId);
    });
    return {
      context: updatedContext,
      items: results,
    };
  });
}
