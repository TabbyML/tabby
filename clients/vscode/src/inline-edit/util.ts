export interface InlineEditParseResult {
  /**
   * mentions, start with '@'
   */
  mentions?: string[];
  /**
   * last mention in the end of user commnad.
   * for `explain @`, mentionQuery is `''`,  we can trigger file pick
   * for `explain @file`, mentionQuery is `file`,  we know user is editing the mention
   * for `explain @file to me`, mentionQuery is `undefined`
   */
  mentionQuery?: string;
}

export const parseUserCommand = (input: string): InlineEditParseResult => {
  const mentions: string[] = [];
  const regex = /(?<=\s|^)@(\S*)/g;
  let match;
  const matches = [];

  while ((match = regex.exec(input)) !== null) {
    const file = match[1];
    if (file) {
      mentions.push(file);
    }
    matches.push(match);
  }

  let mentionQuery = undefined;
  if (matches.length > 0) {
    const lastMatch = matches[matches.length - 1];
    if (lastMatch) {
      const endPos = lastMatch.index + lastMatch[0].length;
      if (endPos === input.length) {
        mentionQuery = lastMatch[1] || "";
      }
    }
  }

  return {
    mentions,
    mentionQuery: mentionQuery !== undefined ? mentionQuery : undefined,
  };
};

export const replaceLastOccurrence = (str: string, substrToReplace: string, replacementStr: string): string => {
  const lastIndex = str.lastIndexOf(substrToReplace);

  if (lastIndex === -1) {
    return str;
  }

  return str.substring(0, lastIndex) + replacementStr + str.substring(lastIndex + substrToReplace.length);
};

export const noop = () => {
  //
};

export class Deferred<T> {
  public resolve: (value: T | PromiseLike<T>) => void = noop;
  public reject: (err?: unknown) => void = noop;
  public readonly promise: Promise<T>;

  constructor() {
    this.promise = new Promise<T>((resolve, reject) => {
      this.resolve = resolve;
      this.reject = reject;
    });
  }
}
