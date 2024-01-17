import path from "path";

import { LRUCache } from "lru-cache";
import * as vscode from "vscode";
import { URI } from "vscode-uri";

import { locationKeyFn } from "../../../../graph/lsp/graph";
import {
  getGraphDocumentSections as defaultGetDocumentSections,
  DocumentSection,
} from "../../../../graph/lsp/sections";
import { getContextRange } from "../../../doc-context-getters";
import { ContextRetriever, ContextRetrieverOptions, ContextSnippet } from "../../../types";
import { createSubscriber } from "../../../utils";
import { baseLanguageId } from "../../utils";
import { isDefined } from "../../../../utils";

interface Section extends DocumentSection {}

interface ActiveDocument {
  uri: URI;
  languageId: string;
  sections: Section[];
  lastRevalidateAt: number;
  lastLines: number;
}

const TEN_MINUTES = 10 * 60 * 1000;

const NUM_OF_CHANGED_LINES_FOR_SECTION_RELOAD = 3;

const MAX_TRACKED_DOCUMENTS = 10;
const MAX_LAST_VISITED_SECTIONS = 10;

const debugSubscriber = createSubscriber<void>();
export const registerDebugListener = debugSubscriber.subscribe.bind(debugSubscriber);

/**
 * Keeps track of document sections a user is navigating to and retrievers the last visited section
 */
export class SectionHistoryRetriever implements ContextRetriever {
  public identifier = "section-history";
  private disposables: vscode.Disposable[] = [];

  // A map of all active documents that are being tracked. We rely on the LRU cache to evict
  // documents that are not being tracked anymore.
  private activeDocuments: LRUCache<string, ActiveDocument> = new LRUCache<string, ActiveDocument>({
    max: MAX_TRACKED_DOCUMENTS,
  });
  // A list of up to ten sections that were being visited last as identifier via their location.
  private lastVisitedSections: vscode.Location[] = [];

  private constructor(
    private window: Pick<
      typeof vscode.window,
      "onDidChangeVisibleTextEditors" | "onDidChangeTextEditorSelection" | "visibleTextEditors"
    > = vscode.window,
    workspace: Pick<typeof vscode.workspace, "onDidChangeTextDocument"> = vscode.workspace,
    private getDocumentSections: typeof defaultGetDocumentSections = defaultGetDocumentSections,
  ) {
    this.disposables.push(window.onDidChangeVisibleTextEditors(this.onDidChangeVisibleTextEditors.bind(this)));
    this.disposables.push(workspace.onDidChangeTextDocument(this.onDidChangeTextDocument.bind(this)));
    this.disposables.push(window.onDidChangeTextEditorSelection(this.onDidChangeTextEditorSelection.bind(this)));
    void this.onDidChangeVisibleTextEditors();
  }

  public static instance: SectionHistoryRetriever | null = null;
  public static createInstance(
    window?: Pick<
      typeof vscode.window,
      "onDidChangeVisibleTextEditors" | "onDidChangeTextEditorSelection" | "visibleTextEditors"
    >,
    workspace?: Pick<typeof vscode.workspace, "onDidChangeTextDocument">,
    getDocumentSections?: typeof defaultGetDocumentSections,
  ): SectionHistoryRetriever {
    if (this.instance) {
      throw new Error("SectionObserver has already been initialized");
    }
    this.instance = new SectionHistoryRetriever(window, workspace, getDocumentSections);
    return this.instance;
  }

  public async retrieve({
    document,
    position,
    docContext,
  }: {
    document: ContextRetrieverOptions["document"];
    position: ContextRetrieverOptions["position"];
    docContext: ContextRetrieverOptions["docContext"];
  }): Promise<ContextSnippet[]> {
    const section = this.getSectionAtPosition(document, position);
    const contextRange = getContextRange(document, docContext);

    function overlapsContextRange(uri: vscode.Uri, range?: { startLine: number; endLine: number }): boolean {
      if (!contextRange || !range || uri.toString() !== document.uri.toString()) {
        return false;
      }

      return contextRange.start.line <= range.startLine && contextRange.end.line >= range.endLine;
    }

    return (
      await Promise.all(
        this.lastVisitedSections
          .map((location) => this.getActiveDocumentAndSectionForLocation(location))
          .filter(isDefined)
          // Remove any sections that are not in the same language as the current document
          .filter(
            ([sectionDocument]) => baseLanguageId(sectionDocument.languageId) === baseLanguageId(document.languageId),
          )
          .map(([, section]) => section)
          // Exclude the current section which should be included already as part of the
          // prefix/suffix.
          .filter(
            (compareSection) =>
              locationKeyFn(compareSection.location) !== (section ? locationKeyFn(section.location) : null),
          )
          // Remove sections that overlap the current prefix/suffix range to avoid
          // duplication.
          .filter(
            (section) =>
              !overlapsContextRange(section.location.uri, {
                startLine: section.location.range.start.line,
                endLine: section.location.range.end.line,
              }),
          )
          // Load the fresh file contents for the sections.
          .map(async (section) => {
            try {
              const uri = section.location.uri;
              const textDocument = await vscode.workspace.openTextDocument(uri);
              const fileName = path.normalize(vscode.workspace.asRelativePath(uri.fsPath));
              const content = textDocument.getText(section.location.range);
              return { fileName, content };
            } catch (error) {
              // Ignore errors opening the text file. This can happen when the file was deleted
              console.error(error);
              return undefined;
            }
          }),
      )
    ).filter(isDefined);
  }

  public isSupportedForLanguageId(): boolean {
    return true;
  }

  private getSectionAtPosition(document: vscode.TextDocument, position: vscode.Position): Section | undefined {
    return this.activeDocuments
      .get(document.uri.toString())
      ?.sections.find((section) => section.location.range.contains(position));
  }

  /**
   * A pretty way to print the current state of all cached sections
   *
   * Printed paths are always in posix format (forwards slashes) even on windows
   * for consistency.
   */
  public debugPrint(selectedDocument?: vscode.TextDocument, selections?: readonly vscode.Selection[]): string {
    const lines: string[] = [];
    this.activeDocuments.forEach((document) => {
      lines.push(path.posix.normalize(vscode.workspace.asRelativePath(document.uri)));
      for (const section of document.sections) {
        const isSelected =
          selectedDocument?.uri.toString() === document.uri.toString() &&
          selections?.some((selection) => section.location.range.contains(selection));
        const isLast = document.sections.at(-1) === section;

        lines.push(`  ${isLast ? "└" : "├"}${isSelected ? "*" : "─"} ` + (section.fuzzyName ?? "unknown"));
      }
    });

    const lastSections = this.lastVisitedSections
      .map((loc) => this.getActiveDocumentAndSectionForLocation(loc)?.[1])
      .filter(isDefined);
    if (lastSections.length > 0) {
      lines.push("");
      lines.push("Last visited sections:");
      for (let i = 0; i < lastSections.length; i++) {
        const section = lastSections[i];
        const isLast = i === lastSections.length - 1;
        const filePath = path.posix.normalize(vscode.workspace.asRelativePath(section.location.uri));

        lines.push(`  ${isLast ? "└" : "├"} ${filePath} ${section.fuzzyName ?? "unknown"}`);
      }
    }

    return lines.join("\n");
  }

  /**
   * Loads or reloads a document's sections and attempts to merge new sections with existing
   * sections.
   *
   * TODO(philipp-spiess): Handle the case that a document is being reloaded while it is still
   * loaded.
   */
  private async loadDocument(document: vscode.TextDocument): Promise<void> {
    const uri = document.uri;
    const lastRevalidateAt = Date.now();
    const lastLines = document.lineCount;
    const sections = (await this.getDocumentSections(document)).map((section) => ({
      ...section,
      lastRevalidateAt,
      lastLines: section.location.range.end.line - section.location.range.start.line,
    }));

    const existingDocument = this.activeDocuments.get(uri.toString());
    if (!existingDocument) {
      this.activeDocuments.set(uri.toString(), {
        uri,
        languageId: document.languageId,
        sections,
        lastRevalidateAt,
        lastLines,
      });
      return;
    }

    // If a document already exists, attempt to diff the sections
    const sectionsToRemove: Section[] = [];
    for (const existingSection of existingDocument.sections) {
      const key = locationKeyFn(existingSection.location);
      const newSection = sections.find((section) => locationKeyFn(section.location) === key);
      if (newSection) {
        existingSection.fuzzyName = newSection.fuzzyName;
        existingSection.location = newSection.location;
      } else {
        sectionsToRemove.push(existingSection);
      }
    }
    for (const sectionToRemove of sectionsToRemove) {
      const index = existingDocument.sections.indexOf(sectionToRemove);
      if (index !== -1) {
        existingDocument.sections.splice(index, 1);
      }
    }
    for (const newSection of sections) {
      const key = locationKeyFn(newSection.location);
      const existingSection = existingDocument.sections.find((section) => locationKeyFn(section.location) === key);
      if (!existingSection) {
        existingDocument.sections.push(newSection);
      }
    }

    debugSubscriber.notify();
  }

  /**
   * Diff vscode.window.visibleTextEditors with activeDocuments to load new documents.
   *
   * We rely on the LRU cache to evict documents that are no longer visible.
   *
   * TODO(philipp-spiess): When this method is called while the documents are still being loaded,
   * we might reload a document immediately afterwards.
   */
  private async onDidChangeVisibleTextEditors(): Promise<void> {
    const promises: Promise<void>[] = [];
    for (const editor of this.window.visibleTextEditors) {
      if (editor.document.uri.scheme !== "file") {
        continue;
      }
      const uri = editor.document.uri.toString();
      if (!this.activeDocuments.has(uri)) {
        promises.push(this.loadDocument(editor.document));
      }
    }
    await Promise.all(promises);
  }

  private getActiveDocumentAndSectionForLocation(location: vscode.Location): [ActiveDocument, Section] | undefined {
    const uri = location.uri.toString();
    if (!this.activeDocuments.has(uri)) {
      return undefined;
    }
    const document = this.activeDocuments.get(uri);
    if (!document) {
      return undefined;
    }
    const locationKey = locationKeyFn(location);
    const section = document.sections.find((section) => locationKeyFn(section.location) === locationKey);
    if (section) {
      return [document, section];
    }
    return undefined;
  }

  private async onDidChangeTextDocument(event: vscode.TextDocumentChangeEvent): Promise<void> {
    const uri = event.document.uri.toString();
    if (!this.activeDocuments.has(uri)) {
      return;
    }

    const document = this.activeDocuments.get(uri)!;

    // We start by checking if the document has changed significantly since sections were last
    // loaded. If so, we reload the document which will mark all sections as dirty.
    const documentChangedSignificantly =
      Math.abs(document.lastLines - event.document.lineCount) >= NUM_OF_CHANGED_LINES_FOR_SECTION_RELOAD;
    const sectionsOutdated = Date.now() - document.lastRevalidateAt > TEN_MINUTES;
    if (documentChangedSignificantly || sectionsOutdated) {
      await this.loadDocument(event.document);
      return;
    }
  }

  /**
   * When the cursor is moving into a tracked selection, we log the access to keep track of
   * frequently visited sections.
   */
  private onDidChangeTextEditorSelection(event: vscode.TextEditorSelectionChangeEvent): void {
    const editor = event.textEditor;
    const position = event.selections[0].active;

    const section = this.getSectionAtPosition(editor.document, position);
    if (!section) {
      return;
    }

    pushUniqueAndTruncate(this.lastVisitedSections, section.location, MAX_LAST_VISITED_SECTIONS);
    debugSubscriber.notify();
  }

  public dispose(): void {
    SectionHistoryRetriever.instance = null;
    for (const disposable of this.disposables) {
      disposable.dispose();
    }
    debugSubscriber.notify();
  }
}

function pushUniqueAndTruncate(array: vscode.Location[], item: vscode.Location, truncate: number): vscode.Location[] {
  const indexOf = array.findIndex((i) => locationKeyFn(i) === locationKeyFn(item));
  if (indexOf > -1) {
    // Remove the item so it is put to the front again
    array.splice(indexOf, 1);
  }
  if (array.length >= truncate) {
    array.pop();
  }
  array.unshift(item);
  return array;
}
