export interface ErrorLike {
  message: string;
  name?: string;
}

export const isErrorLike = (value: unknown): value is ErrorLike =>
  typeof value === "object" && value !== null && ("stack" in value || "message" in value) && !("__typename" in value);

/**
 * Returns true if `val` is not `null` or `undefined`
 */
export const isDefined = <T>(value: T): value is NonNullable<T> => value !== undefined && value !== null;

/**
 * Returns all but the last element of path, or "." if that would be the empty path.
 */
export function dirname(path: string): string {
  return path.split("/").slice(0, -1).join("/") || ".";
}

/**
 * Returns the last element of path, or "." if path is empty.
 */
export function basename(path: string): string {
  return path.split("/").at(-1) || ".";
}

export function pluralize(string: string, count: number | bigint, plural = string + "s"): string {
  return count === 1 || count === 1n ? string : plural;
}

/**
 * Escapes markdown by escaping all ASCII punctuation.
 *
 * Note: this does not escape whitespace, so when rendered markdown will
 * likely collapse adjacent whitespace.
 */
export const escapeMarkdown = (text: string): string => {
  /*
   * GFM you can escape any ASCII punctuation [1]. So we do that, with two
   * special notes:
   * - we escape "\" first to prevent double escaping it
   * - we replace < and > with HTML escape codes to prevent needing to do
   *   HTML escaping.
   * [1]: https://github.github.com/gfm/#backslash-escapes
   */
  const punctuation = "\\!\"#%&'()*+,-./:;=?@[]^_`{|}~";
  for (const char of punctuation) {
    text = text.replaceAll(char, "\\" + char);
  }
  return text.replaceAll("<", "&lt;").replaceAll(">", "&gt;");
};

/**
 * Return a filtered version of the given array, de-duplicating items based on the given key function.
 * The order of the filtered array is not guaranteed to be related to the input ordering.
 */
export const dedupeWith = <T>(items: T[], key: keyof T | ((item: T) => string)): T[] => [
  ...new Map(items.map((item) => [typeof key === "function" ? key(item) : item[key], item])).values(),
];

export const isError = (value: unknown): value is Error => value instanceof Error;

import * as vscode from "vscode";

/**
 * Get the last part of the file path after the last slash
 */
export function getFileNameAfterLastDash(filePath: string): string {
  const lastDashIndex = filePath.lastIndexOf("/");
  if (lastDashIndex === -1) {
    return filePath;
  }
  return filePath.slice(lastDashIndex + 1);
}

export function getEditorInsertSpaces(uri: vscode.Uri): boolean {
  const editor = vscode.window.visibleTextEditors.find((editor) => editor.document.uri === uri);
  if (!editor) {
    // Default to the same as VS Code default
    return true;
  }

  const { languageId } = editor.document;
  const languageConfig = vscode.workspace.getConfiguration(`[${languageId}]`, uri);
  const languageSetting = languageConfig.get("editor.insertSpaces") as boolean | undefined;
  // Prefer language specific setting.
  const insertSpaces = languageSetting || editor.options.insertSpaces;

  // This should never happen: "When getting a text editor's options, this property will always be a boolean (resolved)."
  if (typeof insertSpaces === "string" || insertSpaces === undefined) {
    console.error('Unexpected value when getting "insertSpaces" for the current editor.');
    return true;
  }

  return insertSpaces;
}

export function getEditorTabSize(uri: vscode.Uri): number {
  const editor = vscode.window.visibleTextEditors.find((editor) => editor.document.uri === uri);
  if (!editor) {
    // Default to the same as VS Code default
    return 4;
  }

  const { languageId } = editor.document;
  const languageConfig = vscode.workspace.getConfiguration(`[${languageId}]`, uri);
  const languageSetting = languageConfig.get("editor.tabSize") as number | undefined;
  // Prefer language specific setting.
  const tabSize = languageSetting || editor.options.tabSize;

  // This should never happen: "When getting a text editor's options, this property will always be a number (resolved)."
  if (typeof tabSize === "string" || tabSize === undefined) {
    console.error('Unexpected value when getting "tabSize" for the current editor.');
    return 4;
  }

  return tabSize;
}
