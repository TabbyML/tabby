export const MARKDOWN_SOURCE_REGEX = /\[\[source:\s*([^\]]+)\]\]/g
export const PLACEHOLDER_SOURCE_REGEX = /\[\[source:\s*({.*?})\]\]/g
export const MARKDOWN_CITATION_REGEX = /\[\[citation:\s*(\d+)\]\]/g
export const PLACEHOLDER_FILE_REGEX = /\[\[file:\s*({.*?})\]\]/g
export const MARKDOWN_FILE_REGEX = /\[\[file:\s*([^\]]+)\]\]/g
export const PLACEHOLDER_SYMBOL_REGEX = /\[\[symbol:\s*({.*?})\]\]/g
export const MARKDOWN_SYMBOL_REGEX = /\[\[symbol:\s*([^\]]+)\]\]/g
export const PLACEHOLDER_COMMAND_REGEX = /\[\[contextCommand:\s*(.+?)\]\]/g
export const MARKDOWN_COMMAND_REGEX = /\[\[contextCommand:\s*([^\]]+)\]\]/g

export const DIFF_CHANGES_REGEX = /```diff label=changes[\s\S]*?```/g
