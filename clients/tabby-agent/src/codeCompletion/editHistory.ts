import { TextDocument } from "vscode-languageserver-textdocument";
import { Position, Range, FoldingRange, Connection } from "vscode-languageserver";
import { TextDocuments } from "../lsp/textDocuments";
import type { Configurations } from "../config";
import { getLogger } from "../logger";
import * as diff from "diff";
import type { components as TabbyApiComponents } from "tabby-openapi/compatible";
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
 * Configuration options for window strategy
 */
interface WindowStrategyConfig {
  useFoldingRanges: boolean;
  defaultLinesAround: number;
  minLines: number;
  maxLines: number;
  requireCompleteBrackets: boolean;
}

/**
 * Uniquely identifies a code block within a file
 */
interface BlockIdentifier {
  startLine: number;
  endLine: number;
}

/**
 * Represents a tracked code block with its edit history
 */
interface TrackedBlock {
  originalContent: string;
  edits: string[];
  lastUpdated: number;
}

/**
 * Class responsible for tracking edit history for files
 */
export class EditHistoryTracker {
  private readonly logger = getLogger("EditHistoryTracker");
  private fileContentCache: Map<string, string> = new Map();
  // Map structure: uri -> blockKey -> TrackedBlock
  private trackedBlocksCache: Map<string, Map<string, TrackedBlock>> = new Map();
  private maxEditsPerBlock: number = 10;
  private maxDiffContextSize: number = 5000;
  private windowConfig: WindowStrategyConfig;
  private connection: Connection | null = null;
  private initPromises: Map<string, Promise<void>> = new Map();

  constructor(
    private readonly documents: TextDocuments<TextDocument>,
    private config: Configurations,
    connection?: Connection,
  ) {
    this.connection = connection || null;

    this.windowConfig = {
      useFoldingRanges: true,
      defaultLinesAround: 20,
      minLines: 5,
      maxLines: 100,
      requireCompleteBrackets: true,
    };

    for (const document of documents.all()) {
      this.logger.info(`Initializing edit history for ${document.uri}`);
      this.trackedBlocksCache.set(document.uri, new Map());
      const initPromise = Promise.resolve();
      this.initPromises.set(document.uri, initPromise);
    }

    documents.onDidChangeContent((change) => {
      this.trackDocumentChange(change.document).catch((err) => {
        this.logger.error(`Error tracking document change: ${err}`, err);
      });
    });

    this.updateConfig(config);
  }

  /**
   * Get a string key for a block identifier
   */
  private getBlockKey(block: BlockIdentifier): string {
    return `${block.startLine}-${block.endLine}`;
  }

  /**
   * Track document changes
   */
  private async trackDocumentChange(document: TextDocument): Promise<void> {
    const uri = document.uri;
    const newContent = document.getText();
    const oldContent = this.fileContentCache.get(uri);

    if (this.initPromises.has(uri)) {
      try {
        await this.initPromises.get(uri);
        this.initPromises.delete(uri);
      } catch (error) {
        this.logger.error(`Error waiting for document initialization: ${error}`, error);
      }
    }

    if (oldContent === undefined) {
      this.logger.info(`First time seeing file ${uri}, initializing cache`);
      this.fileContentCache.set(uri, newContent);
      if (!this.trackedBlocksCache.has(uri)) {
        this.trackedBlocksCache.set(uri, new Map());
      }
      return;
    }

    if (oldContent === newContent) {
      return;
    }

    // Update file content cache
    this.fileContentCache.set(uri, newContent);
  }

  /**
   * Get the edit history for a document at a specific position
   */
  public async getEditHistory(uri: string, position: Position): Promise<EditHistory | undefined> {
    if (this.initPromises.has(uri)) {
      try {
        await this.initPromises.get(uri);
        this.initPromises.delete(uri);
      } catch (error) {
        this.logger.error(`Error waiting for document initialization: ${error}`, error);
      }
    }

    const document = this.documents.get(uri);
    if (!document) {
      this.logger.info(`No document available for ${uri}`);
      return undefined;
    }

    try {
      // Get the current context window for the position
      const contextWindow = await this.getSmartContextWindow(document, position);

      // Identify the block
      const blockId = await this.identifyCodeBlock(document, position);
      const blockKey = this.getBlockKey(blockId);

      // Get or create tracked block
      let trackedBlockMap = this.trackedBlocksCache.get(uri);
      if (!trackedBlockMap) {
        trackedBlockMap = new Map();
        this.trackedBlocksCache.set(uri, trackedBlockMap);
      }

      let trackedBlock = trackedBlockMap.get(blockKey);
      if (!trackedBlock) {
        // First time seeing this block, initialize it
        trackedBlock = {
          originalContent: contextWindow,
          edits: [],
          lastUpdated: Date.now(),
        };
        trackedBlockMap.set(blockKey, trackedBlock);
        this.logger.info(`Initialized new tracked block at ${blockKey} for ${uri}`);
      } else {
        // Block exists, calculate diff
        const unifiedDiff = this.generateUnifiedDiff(uri, trackedBlock.originalContent, contextWindow);

        if (trackedBlock.edits.length === 0 || trackedBlock.edits[trackedBlock.edits.length - 1] !== unifiedDiff) {
          trackedBlock.edits.push(unifiedDiff);
          trackedBlock.lastUpdated = Date.now();

          // Cap the edits array size
          if (trackedBlock.edits.length > this.maxEditsPerBlock) {
            trackedBlock.edits.shift();
          }

          this.logger.info(
            `Updated tracked block at ${blockKey} for ${uri}, edits queue size: ${trackedBlock.edits.length}`,
          );
        }
      }

      // Get the most recent diff
      const latestDiff = trackedBlock.edits.length > 0 ? trackedBlock.edits[trackedBlock.edits.length - 1] : "";

      const result: EditHistory = {
        originalCode: trackedBlock.originalContent,
        editsDiff: latestDiff || "",
        currentVersion: {
          content: contextWindow,
          cursorPosition: {
            line: position.line,
            character: position.character,
          },
        },
      };

      this.logger.info(
        `Generated edit history for block ${blockKey} in ${uri} with context window size ${this.countLines(contextWindow)} lines`,
      );
      return result;
    } catch (error) {
      this.logger.error(`Failed to get edit history for ${uri}: ${error}`, error);
      return undefined;
    }
  }

  /**
   * Identify the code block containing the position
   */
  private async identifyCodeBlock(document: TextDocument, position: Position): Promise<BlockIdentifier> {
    if (this.windowConfig.useFoldingRanges && this.connection) {
      try {
        const foldingRanges = (await this.connection.sendRequest("textDocument/foldingRange", {
          textDocument: { uri: document.uri },
        })) as FoldingRange[];

        if (foldingRanges && foldingRanges.length > 0) {
          const containingRange = this.findSmallestContainingRange(foldingRanges, position);

          if (containingRange) {
            return {
              startLine: containingRange.startLine,
              endLine: containingRange.endLine,
            };
          }
        }
      } catch (error) {
        this.logger.error(`Error getting folding ranges: ${error}`, error);
      }
    }

    // Fallback to fixed window
    return {
      startLine: Math.max(0, position.line - this.windowConfig.defaultLinesAround),
      endLine: position.line + this.windowConfig.defaultLinesAround,
    };
  }

  /**
   * Reset the original content to the current version after a NES request
   * This ensures that subsequent edits are compared against the accepted state
   */
  public async updateOriginalContentToCurrentVersion(uri: string, position: Position): Promise<void> {
    if (this.initPromises.has(uri)) {
      try {
        await this.initPromises.get(uri);
        this.initPromises.delete(uri);
      } catch (error) {
        this.logger.error(`Error waiting for document initialization: ${error}`, error);
        return;
      }
    }

    const document = this.documents.get(uri);
    if (!document) {
      this.logger.info(`No document available for ${uri}`);
      return;
    }

    try {
      // Get the current context window
      const contextWindow = await this.getSmartContextWindow(document, position);

      // Identify the block
      const blockId = await this.identifyCodeBlock(document, position);
      const blockKey = this.getBlockKey(blockId);

      // Get or create tracked block map
      let trackedBlockMap = this.trackedBlocksCache.get(uri);
      if (!trackedBlockMap) {
        trackedBlockMap = new Map();
        this.trackedBlocksCache.set(uri, trackedBlockMap);
      }

      // Update the original content for this block
      const trackedBlock = {
        originalContent: contextWindow,
        edits: [],
        lastUpdated: Date.now(),
      };
      trackedBlockMap.set(blockKey, trackedBlock);

      this.logger.info(`Reset original content for block ${blockKey} in ${uri} to current version`);
    } catch (error) {
      this.logger.error(`Failed to update original content to current version for ${uri}: ${error}`, error);
    }
  }

  /**
   * Generate a unified diff between old and new content
   */
  private generateUnifiedDiff(uri: string, oldContent: string, newContent: string): string {
    const filename = uri.split("/").pop() || uri;

    const patch = diff
      .createPatch(filename, oldContent, newContent, undefined, undefined, {
        context: 0,
      })
      .split("\n")
      .slice(2)
      .join("\n");

    if (patch.length > this.maxDiffContextSize) {
      this.logger.warn(`Diff for ${uri} exceeds max size (${patch.length} > ${this.maxDiffContextSize}), truncating`);
      return patch.substring(0, this.maxDiffContextSize);
    }

    return patch;
  }

  /**
   * Get a smart context window around a position using folding ranges
   */
  private async getSmartContextWindow(document: TextDocument, position: Position): Promise<string> {
    if (this.windowConfig.useFoldingRanges && this.connection) {
      try {
        this.logger.info(`Attempting to get folding ranges for ${document.uri}`);

        const foldingRanges = (await this.connection.sendRequest("textDocument/foldingRange", {
          textDocument: { uri: document.uri },
        })) as FoldingRange[];

        if (foldingRanges && foldingRanges.length > 0) {
          const containingRange = this.findSmallestContainingRange(foldingRanges, position);

          if (containingRange) {
            const lines = document.getText().split("\n");
            const endLine = Math.min(containingRange.endLine + 1, lines.length);

            const range = Range.create(containingRange.startLine, 0, endLine, 0);

            const contextWindow = this.getTextForRange(document, range);
            const lineCount = this.countLines(contextWindow);

            this.logger.info(`Found containing folding range with ${lineCount} lines`);

            if (this.isContextSufficient(contextWindow)) {
              return contextWindow;
            } else {
              this.logger.info(`Folding range context window not sufficient, falling back to fixed line window`);
            }
          } else {
            this.logger.info(`No containing folding range found for position line ${position.line}`);
          }
        } else {
          this.logger.info(`No folding ranges returned for ${document.uri}`);
        }
      } catch (error) {
        this.logger.error(`Error getting folding ranges: ${error}`, error);
      }
    }

    return this.getFixedLineWindow(document, position);
  }

  /**
   * Find the smallest folding range that contains the position
   */
  private findSmallestContainingRange(ranges: FoldingRange[], position: Position): FoldingRange | null {
    let smallestRange: FoldingRange | null = null;
    let smallestSize = Infinity;

    for (const range of ranges) {
      if (range.startLine <= position.line && range.endLine >= position.line) {
        const size = range.endLine - range.startLine + 1;

        if (size < smallestSize) {
          smallestRange = range;
          smallestSize = size;
        }
      }
    }

    return smallestRange;
  }

  /**
   * Get text for a specific range in a document
   */
  private getTextForRange(document: TextDocument, range: Range): string {
    const content = document.getText();
    const lines = content.split("\n");

    const startLine = Math.max(0, Math.min(range.start.line, lines.length - 1));
    const endLine = Math.max(0, Math.min(range.end.line, lines.length - 1));

    const linesInRange = lines.slice(startLine, endLine + 1);

    return linesInRange.join("\n");
  }

  /**
   * Get a fixed number of lines around a position
   */
  private getFixedLineWindow(document: TextDocument, position: Position): string {
    const content = document.getText();
    const lines = content.split("\n");

    const startLine = Math.max(0, position.line - this.windowConfig.defaultLinesAround);
    const endLine = Math.min(lines.length - 1, position.line + this.windowConfig.defaultLinesAround);

    const linesInRange = lines.slice(startLine, endLine + 1);

    this.logger.info(
      `Generated fixed line window with ${linesInRange.length} lines around position line ${position.line}`,
    );

    return linesInRange.join("\n");
  }

  /**
   * Check if a context window is sufficient based on configuration
   */
  private isContextSufficient(context: string): boolean {
    const lineCount = this.countLines(context);
    if (lineCount < this.windowConfig.minLines) {
      return false;
    }

    if (lineCount > this.windowConfig.maxLines) {
      return false;
    }

    if (this.windowConfig.requireCompleteBrackets && !this.hasBalancedBrackets(context)) {
      return false;
    }

    return true;
  }

  /**
   * Count the number of lines in a string
   */
  private countLines(text: string): number {
    return (text.match(/\n/g) || []).length + 1;
  }

  /**
   * Check if a string has balanced brackets
   * This is a simplified check that accepts more code fragments
   */
  private hasBalancedBrackets(text: string): boolean {
    const stack: string[] = [];

    for (let i = 0; i < text.length; i++) {
      const char = text[i];
      if (char === "{") {
        stack.push(char);
      } else if (char === "}") {
        if (stack.length === 0 || stack.pop() !== "{") {
          return false;
        }
      }
    }

    return true;
  }

  /**
   * Find a position where the text differs between old and new content
   */
  private findChangePosition(oldContent: string, newContent: string): Position {
    if (oldContent.length > 1000000 || newContent.length > 1000000) {
      this.logger.info("Large file detected, using optimized change detection");

      const lengthDiff = newContent.length - oldContent.length;
      const estimatedChangePct = Math.abs(lengthDiff) / Math.max(oldContent.length, newContent.length);

      if (estimatedChangePct < 0.05) {
        let line = 0;
        let char = 0;

        const minLength = Math.min(oldContent.length, newContent.length);
        for (let i = 0; i < minLength; i++) {
          if (oldContent[i] !== newContent[i]) {
            const textBefore = oldContent.substring(0, i);
            const lines = textBefore.split("\n");
            line = lines.length - 1;
            char = (lines[lines.length - 1] || "").length;
            break;
          }
        }

        return Position.create(line, char);
      }
    }

    const oldLines = oldContent.split("\n");
    const newLines = newContent.split("\n");

    for (let i = 0; i < Math.max(oldLines.length, newLines.length); i++) {
      if (oldLines[i] !== newLines[i]) {
        const oldLine = oldLines[i] || "";
        const newLine = newLines[i] || "";

        for (let j = 0; j < Math.max(oldLine.length, newLine.length); j++) {
          if (oldLine[j] !== newLine[j]) {
            return Position.create(i, j);
          }
        }

        return Position.create(i, Math.min(oldLine.length, newLine.length));
      }
    }

    return Position.create(0, 0);
  }

  /**
   * Update the tracker's configuration
   */
  public updateConfig(config: Configurations): void {
    const mergedConfig = config.getMergedConfig();

    const nextEditConfig = mergedConfig.completion.nextEditSuggestion || {
      enabled: true,
      maxEditsPerFile: 10,
      maxDiffContextSize: 5000,
    };

    this.maxEditsPerBlock = nextEditConfig.maxEditsPerFile || 10;
    this.maxDiffContextSize = nextEditConfig.maxDiffContextSize || 5000;

    this.windowConfig = {
      useFoldingRanges: true,
      defaultLinesAround: 20,
      minLines: 5,
      maxLines: 100,
      requireCompleteBrackets: true,
    };

    this.logger.info(
      `Updated config: useFoldingRanges=${this.windowConfig.useFoldingRanges}, ` +
        `defaultLinesAround=${this.windowConfig.defaultLinesAround}, ` +
        `maxEditsPerBlock=${this.maxEditsPerBlock}, ` +
        `maxDiffContextSize=${this.maxDiffContextSize}`,
    );
  }

  /**
   * Clear the edit history for a specific file or all files
   */
  public clearHistory(uri?: string): void {
    if (uri) {
      this.trackedBlocksCache.delete(uri);
      this.logger.info(`Cleared edit history for ${uri}`);
    } else {
      this.trackedBlocksCache.clear();
      this.logger.info("Cleared all edit history");
    }
  }
}

/**
 * Build edit history for next edit suggestion mode
 * Converts from camelCase to snake_case format for the API
 */
export function buildEditHistoryForRequest(
  editHistory: EditHistory,
): TabbyApiComponents["schemas"]["Segments"]["edit_history"] {
  const originalCode = editHistory.originalCode;
  const editsDiff = editHistory.editsDiff;
  const currentVersionContent = editHistory.currentVersion.content;
  const cursorPosition = {
    line: editHistory.currentVersion.cursorPosition.line,
    character: editHistory.currentVersion.cursorPosition.character,
  };

  return {
    original_code: originalCode,
    edits_diff: editsDiff,
    current_version: {
      content: currentVersionContent,
      cursor_position: cursorPosition,
    },
  };
}
