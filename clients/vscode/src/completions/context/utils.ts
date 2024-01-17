/**
 * Returns the base language id for the given language id. This is used to determine which language
 * IDs can be included as context for a given language ID.
 *
 * TODO(beyang): handle JavaScript <-> TypeScript and verify this works for C header files omit
 * files of other languages
 */
export function baseLanguageId(languageId: string): string {
  switch (languageId) {
    case "typescript":
    case "typescriptreact":
    case "javascript":
    case "javascriptreact":
      return "typescriptreact";
    default:
      return languageId;
  }
}

// On Node.js, an abort controller is implemented with the default event emitter
// helpers which will warn when you add more than 10 event listeners. Since the
// graph context implementation can fan out to many more concurrent requests, we
// easily reach that limit causing a lot of noise in the console. This is a
// lightweight abort controller implementation that does not have that limit.
export class CustomAbortController {
  public signal = new CustomAbortSignal();
  public abort(): void {
    this.signal.abort();
  }
}
export class CustomAbortSignal {
  private listeners: Set<() => void> = new Set();
  public aborted = false;

  public addEventListener(_eventName: "abort", listener: () => void): void {
    if (this.aborted) {
      void Promise.resolve().then(() => listener());
      return;
    }
    this.listeners.add(listener);
  }

  public removeEventListener(listener: () => void): void {
    this.listeners.delete(listener);
  }

  public abort(): void {
    if (this.aborted) {
      return;
    }
    this.aborted = true;
    for (const listener of this.listeners) {
      listener();
    }
    this.listeners.clear();
  }
}
