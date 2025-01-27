export interface InlineChatParseResult {
  command: string;
  mentions?: string[];
  mentionQuery?: string;
}

export const parseInput = (input: string): InlineChatParseResult => {
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

  const command = input.replace(regex, "").trim();

  return {
    command,
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
