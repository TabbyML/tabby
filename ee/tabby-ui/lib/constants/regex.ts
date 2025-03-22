export const MARKDOWN_SOURCE_REGEX_ESCAPED =
  /(?:\\?\[\\?\[|\[\[)source:\s*([^\]]+)(?:\\?\]\\?\]|\]\])/g

export const MARKDOWN_CITATION_REGEX_ESCAPED =
  /(?:\\?\[\\?\[|\[\[)?citation:\s*(\d+)(?:\\?\]\\?\]|\]\])?/g

export const PLACEHOLDER_FILE_REGEX_ESCAPED =
  /(?:\\?\[\\?\[|\[\[)file:\s*({.*?})(?:\\?\]\\?\]|\]\])/g

export const MARKDOWN_FILE_REGEX_ESCAPED =
  /(?:\\?\[\\?\[|\[\[)file:\s*([^\]]+)(?:\\?\]\\?\]|\]\])/g

export const PLACEHOLDER_SYMBOL_REGEX_ESCAPED =
  /(?:\\?\[\\?\[|\[\[)symbol:\s*({.*?})(?:\\?\]\\?\]|\]\])/g

export const MARKDOWN_SYMBOL_REGEX_ESCAPED =
  /(?:\\?\[\\?\[|\[\[)symbol:\s*([^\]]+)(?:\\?\]\\?\]|\]\])/g

export const PLACEHOLDER_COMMAND_REGEX_ESCAPED =
  /(?:\\?\[\\?\[|\[\[)contextCommand:\s*"(.+?)"(?:\\?\]\\?\]|\]\])/g

export const MARKDOWN_COMMAND_REGEX_ESCAPED =
  /(?:\\?\[\\?\[|\[\[)contextCommand:\s*([^\]]+)(?:\\?\]\\?\]|\]\])/g
