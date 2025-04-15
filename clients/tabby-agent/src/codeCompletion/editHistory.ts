import { TextDocument } from "vscode-languageserver-textdocument";
import { Position } from "vscode-languageserver";
import { TextDocuments } from "../lsp/textDocuments";
import type { Configurations } from "../config";
import { getLogger } from "../logger";
import * as diff from "diff";

/**
 * Interface for the cursor position in a document
 */
export interface CursorPosition {
  line: number;
  character: number;
}

/**
 * Interface for the current version of a document
 */
export interface CurrentVersion {
  content: string;
  cursorPosition: CursorPosition;
}

/**
 * Interface for the edit history of a document
 */
export interface EditHistory {
  originalCode: string;
  editsDiff: string;
  currentVersion: CurrentVersion;
}

/**
 * Class responsible for tracking edit history for files
 */
export class EditHistoryTracker {
  private readonly logger = getLogger("EditHistoryTracker");
  private fileContentCache: Map<string, string> = new Map();
  private editsQueue: Map<string, string[]> = new Map();
  private maxEditsPerFile: number = 10;
  private maxDiffContextSize: number = 5000;

  constructor(
    private readonly documents: TextDocuments<TextDocument>,
    private config: Configurations,
  ) {
    // Initialize with current document contents
    for (const document of documents.all()) {
      this.logger.debug(`Initializing edit history for ${document.uri}`);
      this.cacheFileContent(document.uri, document.getText());
    }

    // Listen for document changes
    documents.onDidChangeContent((change) => {
      this.trackDocumentChange(change.document);
    });

    // Update configuration
    this.updateConfig(config);
  }

  /**
   * Cache file content for a document
   */
  private cacheFileContent(uri: string, content: string): void {
    this.fileContentCache.set(uri, content);
  }

  /**
   * Track document changes
   */
  private trackDocumentChange(document: TextDocument): void {
    const uri = document.uri;
    const newContent = document.getText();
    const oldContent = this.fileContentCache.get(uri);

    if (oldContent === undefined) {
      // First time seeing this file
      this.logger.debug(`First time seeing file ${uri}, caching content`);
      this.cacheFileContent(uri, newContent);
      return;
    }

    if (oldContent === newContent) {
      return; // No changes
    }

    // Generate unified diff
    const unifiedDiff = this.generateUnifiedDiff(uri, oldContent, newContent);

    // Add to edits queue with size limit
    if (!this.editsQueue.has(uri)) {
      this.editsQueue.set(uri, []);
    }

    const fileEdits = this.editsQueue.get(uri)!;
    fileEdits.push(unifiedDiff);

    // Maintain max history size
    if (fileEdits.length > this.maxEditsPerFile) {
      fileEdits.shift(); // Remove oldest edit
    }

    // Update cache with new content
    this.cacheFileContent(uri, newContent);
    this.logger.debug(`Tracked document change for ${uri}, edits queue size: ${fileEdits.length}`);
  }

  /**
   * Generate a unified diff between old and new content
   */
  private generateUnifiedDiff(uri: string, oldContent: string, newContent: string): string {
    // Extract file name from URI
    const filename = uri.split("/").pop() || uri;

    // Create a unified diff using the diff library
    const patch = diff.createPatch(filename, oldContent, newContent, "old", "new");

    // Limit the size of the diff if needed
    if (patch.length > this.maxDiffContextSize) {
      this.logger.warn(`Diff for ${uri} exceeds max size (${patch.length} > ${this.maxDiffContextSize}), truncating`);
      return patch.substring(0, this.maxDiffContextSize);
    }

    return patch;
  }

  /**
   * Get the edit history for a document at a specific position
   */
  public getEditHistory(uri: string, position: Position): EditHistory | undefined {
    const originalContent = this.getOriginalContent(uri);
    if (originalContent === undefined) {
      this.logger.debug(`No original content available for ${uri}`);
      return undefined;
    }

    const currentContent = this.documents.get(uri)?.getText();
    if (!currentContent) {
      this.logger.debug(`No current content available for ${uri}`);
      return undefined;
    }

    const edits = this.editsQueue.get(uri) || [];
    const combinedDiff = edits.join("\n");

    return {
      originalCode: originalContent,
      editsDiff: combinedDiff,
      currentVersion: {
        content: currentContent,
        cursorPosition: {
          line: position.line,
          character: position.character,
        },
      },
    };
  }

  /**
   * Get the original content for a document
   */
  private getOriginalContent(uri: string): string | undefined {
    const edits = this.editsQueue.get(uri);
    if (!edits || edits.length === 0) {
      // If we don't have any edits, the current content is the original content
      return this.fileContentCache.get(uri);
    }

    // In a real implementation, we'd reconstruct the original content by applying the reverse of all diffs
    // Here, we'll use a simplified approach: if we have edits, use the current content as a fallback
    // A more accurate implementation would reconstruct from the diffs
    return this.fileContentCache.get(uri);
  }

  /**
   * Update the tracker's configuration
   */
  public updateConfig(config: Configurations): void {
    const mergedConfig = config.getMergedConfig();
    this.maxEditsPerFile = mergedConfig.completion.nextEditSuggestion?.maxEditsPerFile || 10;
    this.maxDiffContextSize = mergedConfig.completion.nextEditSuggestion?.maxDiffContextSize || 5000;
    this.logger.debug(
      `Updated config: maxEditsPerFile=${this.maxEditsPerFile}, maxDiffContextSize=${this.maxDiffContextSize}`,
    );
  }

  /**
   * Clear the edit history for a specific file or all files
   */
  public clearHistory(uri?: string): void {
    if (uri) {
      this.editsQueue.delete(uri);
      this.logger.debug(`Cleared edit history for ${uri}`);
    } else {
      this.editsQueue.clear();
      this.logger.debug("Cleared all edit history");
    }
  }
}

/**
 * Build edit history for next edit suggestion mode
 * Converts from camelCase to snake_case format for the API
 */
export function buildEditHistoryForRequest(editHistory: EditHistory): Record<string, unknown> {
  // Extract values from camelCase properties
  const originalCode = editHistory.originalCode;
  const editsDiff = editHistory.editsDiff;
  const currentVersionContent = editHistory.currentVersion.content;
  const cursorPosition = {
    line: editHistory.currentVersion.cursorPosition.line,
    character: editHistory.currentVersion.cursorPosition.character,
  };

  // Return the object with snake_case keys for the API
  return {
    original_code: originalCode,
    edits_diff: editsDiff,
    current_version: {
      content: currentVersionContent,
      cursor_position: cursorPosition,
    },
  };
}
