export enum MentionType {
  File = "file",
  Symbol = "symbol",
}

export interface Mention {
  /**
   * The text of the mention (without the @ prefix)
   */
  text: string;
  /**
   * The type of the mention
   */
  type: MentionType;
}

export interface InlineEditParseResult {
  /**
   * mentions, start with '@'
   */
  mentions?: Mention[];
  /**
   * last mention in the end of user commnad.
   * for `explain @`, mentionQuery is `''`,  we can trigger pick
   * for `explain @file`, mentionQuery is `file`,  we know user is editing the mention
   * for `explain @file to me`, mentionQuery is `undefined`
   */
  mentionQuery?: string;
}

export const parseUserCommand = (input: string): InlineEditParseResult => {
  const mentions: Mention[] = [];
  // Match @text (both file and symbol mentions use the same @ prefix)
  const regex = /(?<=\s|^)@(\S*)/g;
  let match;
  const matches = [];

  while ((match = regex.exec(input)) !== null) {
    const text = match[1];
    if (text) {
      mentions.push({
        text,
        // Default to File type, will be updated when the mention is resolved
        type: MentionType.File,
      });
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
