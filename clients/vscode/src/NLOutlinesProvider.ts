import {
  CancellationToken,
  CodeLens,
  CodeLensProvider,
  EventEmitter,
  Location,
  Position,
  ProviderResult,
  Range,
  TextDocument,
  TextEditor,
  TextEditorDecorationType,
  Uri,
  window,
  workspace,
  WorkspaceEdit,
} from "vscode";
import { Config } from "./Config";
import OpenAI from "openai";
import generateNLOutlinesPrompt from "../assets/prompts/generateNLOutlines.txt";
import editNLOutline from "../assets/prompts/editNLOutline.txt";
import * as Diff from "diff";

interface ChatNLOutlinesParams {
  location: Location;
  editor?: TextEditor;
}

interface Outline {
  startLine: number;
  endLine: number;
  content: string;
}

interface CodeChangeRequest {
  oldOutline: string;
  oldCode: string;
  newOutline: string;
}

interface PendingChange {
  oldLines: string[];
  newLines: string[];
  decorations: { added: Range[]; removed: Range[] };
  editRange: Range;
  newContent: string;
  originalStartLine: number;
}

interface ChangesPreview {
  oldLines: string[];
  newLines: string[];
  decorations: { added: Range[]; removed: Range[] };
  editRange: Range;
}

type OpenAIResponse = AsyncIterable<OpenAI.Chat.Completions.ChatCompletionChunk>;

export class NLOutlinesProvider extends EventEmitter<void> implements CodeLensProvider {
  private client: OpenAI;
  private outlines: Map<string, Outline[]>;
  private addedDecorationType: TextEditorDecorationType;
  private removedDecorationType: TextEditorDecorationType;
  private pendingChanges: Map<string, PendingChange>;
  private pendingCodeLenses: Map<string, CodeLens[]>;

  constructor(config: Config) {
    super();
    this.client = new OpenAI({
      apiKey: config.serverToken,
      baseURL: config.serverEndpoint + "/v1",
    });
    this.outlines = new Map();
    this.pendingChanges = new Map();
    this.pendingCodeLenses = new Map();
    this.addedDecorationType = window.createTextEditorDecorationType({
      backgroundColor: "rgba(0, 255, 0, 0.2)",
      isWholeLine: true,
    });
    this.removedDecorationType = window.createTextEditorDecorationType({
      backgroundColor: "rgba(255, 0, 0, 0.2)",
      isWholeLine: true,
    });

    window.onDidChangeActiveTextEditor(() => {
      this.clearAllPendingChanges();
    });
  }

  async provideNLOutlinesGenerate(params: ChatNLOutlinesParams): Promise<boolean> {
    if (!params.editor) {
      return false;
    }

    const document = params.editor.document;

    try {
      const selection = new Range(params.location.range.start, params.location.range.end);
      if (selection.isEmpty) {
        throw new Error("No document selected");
      }

      const selectedText = document.getText(selection);
      if (selectedText.length > 3000) {
        throw new Error("Document too long");
      }

      const lines = selectedText.split("\n");
      const startLine = selection.start.line;
      const numberedText = lines.map((line, index) => `${startLine + index + 1} | ${line}`).join("\n");

      const stream = await this.generateNLOutlinesRequest(numberedText);

      let buffer = "";
      const documentOutlines: Outline[] = [];

      for await (const chunk of stream) {
        const content = chunk.choices[0]?.delta?.content || "";
        buffer += content;
        let newlineIndex: number;

        while ((newlineIndex = buffer.indexOf("\n")) !== -1) {
          const fullLine = buffer.slice(0, newlineIndex).trim();
          buffer = buffer.slice(newlineIndex + 1);
          const match = fullLine.match(/^(\d+)\s*\|\s*(\d+)\s*\|\s*(.*)$/);

          if (match) {
            const [, startLineNumber, endLineNumber, content] = match;
            if (!startLineNumber || !endLineNumber) continue;
            const parsedStartLine = parseInt(startLineNumber, 10);
            const parsedEndLine = parseInt(endLineNumber, 10);
            if (!isNaN(parsedStartLine) && !isNaN(parsedEndLine) && content) {
              documentOutlines.push({
                startLine: parsedStartLine - 1,
                endLine: parsedEndLine - 1,
                content,
              });
              this.outlines.set(document.uri.toString(), documentOutlines);
              this.fire();
            }
          }
        }
      }

      this.outlines.set(document.uri.toString(), documentOutlines);
      this.fire();

      return true;
    } catch (error) {
      window.showErrorMessage(`Error generating outlines: ${error instanceof Error ? error.message : String(error)}`);
      return false;
    }
  }

  async updateNLOutline(documentUri: string, lineNumber: number, newContent: string): Promise<boolean> {
    const outlines = this.outlines.get(documentUri) || [];
    const oldOutline = outlines.find((outline) => outline.startLine === lineNumber);
    if (!oldOutline) {
      throw new Error("No matching outline found for the given line number");
    }

    const document = await workspace.openTextDocument(Uri.parse(documentUri));
    const oldCodeRange = new Range(oldOutline.startLine, 0, oldOutline.endLine + 1, 0);
    const oldCode = document.getText(oldCodeRange);
    const oldCodeWithLineNumbers = this.formatCodeWithLineNumbers(oldCode, oldOutline.startLine);

    const changeRequest: CodeChangeRequest = {
      oldOutline: oldOutline.content,
      oldCode: oldCodeWithLineNumbers,
      newOutline: newContent,
    };

    try {
      const stream = await this.generateNewCodeBaseOnEditedRequest(changeRequest);
      let updatedCode = "";
      for await (const chunk of stream) {
        updatedCode += chunk.choices[0]?.delta?.content || "";
      }
      updatedCode = updatedCode.replace(/<GENERATEDCODE>\n?/, "").replace(/\n?<\/GENERATEDCODE>/, "");

      const lines = updatedCode.split("\n").map((line) => {
        const parts = line.split("|");
        if (parts.length > 1) {
          const leftWhitespace = line.match(/^\s*/)?.[0] || "";
          const processedPart = parts.slice(1).join("|").trimEnd();
          return leftWhitespace + processedPart;
        }
        return line.trimEnd();
      });

      if (lines[lines.length - 1] === "") {
        lines.pop();
      }

      const { oldLines, newLines, decorations, editRange } = this.generateChangesPreview(
        oldCode,
        lines.join("\n"),
        oldOutline.startLine,
      );

      this.pendingChanges.set(documentUri, {
        oldLines,
        newLines,
        decorations,
        editRange,
        newContent,
        originalStartLine: oldOutline.startLine,
      });

      const edit = new WorkspaceEdit();
      edit.replace(Uri.parse(documentUri), oldCodeRange, newLines.join("\n"));
      await workspace.applyEdit(edit);

      const editor = window.activeTextEditor;
      if (editor && editor.document.uri.toString() === documentUri) {
        this.applyDecorations(editor, decorations);
        this.addAcceptDiscardCodeLens(editor, editRange, newContent);
      }

      this.fire();
      return true;
    } catch (error) {
      window.showErrorMessage(`Error updating NL Outline: ${error instanceof Error ? error.message : String(error)}`);
      return false;
    }
  }

  private async generateNLOutlinesRequest(documentation: string): Promise<OpenAIResponse> {
    const promptTemplate = editNLOutline;
    const content = promptTemplate.replace("{{document}}", documentation);
    return this.openAIRequest(content);
  }

  private async generateNewCodeBaseOnEditedRequest(changeRequest: CodeChangeRequest): Promise<OpenAIResponse> {
    const promptTemplate = generateNLOutlinesPrompt;
    const changeJson = JSON.stringify(changeRequest, null, 2);
    const content = promptTemplate.replace("{{document}}", changeJson);
    return this.openAIRequest(content);
  }

  private async openAIRequest(question: string): Promise<OpenAIResponse> {
    const messages = [{ role: "user" as const, content: question }];
    return await this.client.chat.completions.create({
      model: "",
      messages: messages,
      stream: true,
    });
  }

  provideCodeLenses(document: TextDocument, token: CancellationToken): ProviderResult<CodeLens[]> {
    if (token.isCancellationRequested) {
      return [];
    }

    const pendingCodeLenses = this.pendingCodeLenses.get(document.uri.toString());
    if (pendingCodeLenses) {
      return pendingCodeLenses;
    }

    const documentOutlines = this.outlines.get(document.uri.toString());
    if (!documentOutlines) {
      return [];
    }

    return documentOutlines.flatMap((outline) => {
      const range = document.lineAt(outline.startLine).range;
      return [
        new CodeLens(range, {
          title: "Edit",
          command: "tabby.chat.edit.editNLOutline",
          arguments: [document.uri, outline.startLine],
        }),
        new CodeLens(range, {
          title: outline.content,
          command: "",
          arguments: [],
        }),
      ];
    });
  }

  resolveCodeLens?(codeLens: CodeLens, token: CancellationToken): ProviderResult<CodeLens> {
    if (token.isCancellationRequested) {
      return codeLens;
    }
    return codeLens;
  }

  get onDidChangeCodeLenses() {
    return this.event;
  }

  clearOutlines(documentUri: string) {
    this.outlines.delete(documentUri);
    this.fire();
  }

  getOutline(documentUri: string, lineNumber: number): string | undefined {
    return this.outlines.get(documentUri)?.find((outline) => outline.startLine === lineNumber)?.content;
  }

  private formatCodeWithLineNumbers(code: string, startLine: number): string {
    return code
      .split("\n")
      .map((line, index) => `${startLine + index} | ${line}`)
      .join("\n");
  }

  private generateChangesPreview(oldCode: string, newCode: string, startLine: number): ChangesPreview {
    const oldLines = oldCode.split("\n");
    const decorations: { added: Range[]; removed: Range[] } = { added: [], removed: [] };
    const added: string[] = [];
    const removed: string[] = [];
    const unchanged: string[] = [];

    Diff.diffLines(oldCode, newCode).forEach((diff) => {
      const lines = diff.value.split("\n").filter((line) => line !== "");
      if (diff.added) {
        added.push(...lines);
      } else if (diff.removed) {
        removed.push(...lines);
      } else {
        unchanged.push(...lines);
      }
    });

    const finalLines = [...added, ...unchanged, ...removed];

    const lastRemovedIndex = finalLines.findLastIndex((line) => removed.includes(line));
    if (lastRemovedIndex !== -1) {
      finalLines.splice(lastRemovedIndex + 1, 0, "");
    }

    let currentLine = startLine;
    finalLines.forEach((line) => {
      if (added.includes(line)) {
        decorations.added.push(new Range(currentLine, 0, currentLine, line.length));
      } else if (removed.includes(line)) {
        decorations.removed.push(new Range(currentLine, 0, currentLine, line.length));
      }
      currentLine++;
    });

    const editRange = new Range(
      startLine,
      0,
      startLine + finalLines.length - 1,
      finalLines[finalLines.length - 1]?.length || 0,
    );

    return { oldLines, newLines: finalLines, decorations, editRange };
  }

  private applyDecorations(editor: TextEditor, decorations: { added: Range[]; removed: Range[] }) {
    editor.setDecorations(this.addedDecorationType, decorations.added);
    editor.setDecorations(this.removedDecorationType, decorations.removed);
  }

  private addAcceptDiscardCodeLens(editor: TextEditor, editRange: Range, newOutline: string) {
    const codeLenses = [
      new CodeLens(editRange, {
        title: "Accept",
        command: "tabby.chat.edit.outline.accept",
      }),
      new CodeLens(editRange, {
        title: "Discard",
        command: "tabby.chat.edit.outline.discard",
      }),
      new CodeLens(editRange, {
        title: newOutline,
        command: "",
        arguments: [],
      }),
    ];

    this.pendingCodeLenses.set(editor.document.uri.toString(), codeLenses);
  }

  async acceptChanges(documentUri: Uri, newOutline: string, originalStartLine: number) {
    const pendingChange = this.pendingChanges.get(documentUri.toString());

    if (pendingChange) {
      const { oldLines, newLines } = pendingChange;

      const startLine = pendingChange.originalStartLine;
      const edit = new WorkspaceEdit();

      const oldLinesCount = oldLines.length;
      const newLinesCount = newLines.length;
      const startDeleteLine = startLine + (newLinesCount - oldLinesCount);
      const endDeleteLine = startLine + newLinesCount - 1;
      const deleteRange = new Range(new Position(startDeleteLine, 0), new Position(endDeleteLine, 0));
      edit.delete(documentUri, deleteRange);

      await workspace.applyEdit(edit);

      const outlines = this.outlines.get(documentUri.toString()) || [];
      const outlineIndex = outlines.findIndex((o) => o.startLine === originalStartLine);

      if (outlineIndex !== -1) {
        outlines[outlineIndex] = {
          startLine: originalStartLine,
          endLine: originalStartLine + newLines.length - 1,
          content: newOutline,
        };

        const lineDifference = newLines.length - oldLines.length;

        for (let i = outlineIndex + 1; i < outlines.length; i++) {
          const outline = outlines[i];
          if (outline) {
            outline.startLine += lineDifference;
            outline.endLine += lineDifference;
          }
        }

        this.outlines.set(documentUri.toString(), outlines);
      }

      await this.clearPendingChanges(documentUri.toString(), true);
      this.fire();
    }
  }

  async discardChanges(documentUri: Uri) {
    const pendingChange = this.pendingChanges.get(documentUri.toString());
    if (pendingChange) {
      const { oldLines, newLines } = pendingChange;
      const startLine = pendingChange.originalStartLine;

      const edit = new WorkspaceEdit();

      const oldLinesCount = oldLines.length;
      const newLinesCount = newLines.length;
      const startDeleteLine = startLine;
      const endDeleteLine = startLine + (newLinesCount - oldLinesCount);
      const deleteRange = new Range(new Position(startDeleteLine, 0), new Position(endDeleteLine, 0));
      edit.delete(documentUri, deleteRange);
      await workspace.applyEdit(edit);
      this.clearPendingChanges(documentUri.toString(), true);
      this.fire();
    }
  }

  private async clearPendingChanges(documentUri: string, isDeleted: boolean) {
    const pendingChange = this.pendingChanges.get(documentUri.toString());
    const editor = window.activeTextEditor;

    if (pendingChange && !isDeleted) {
      const { oldLines, newLines } = pendingChange;
      const startLine = pendingChange.originalStartLine;

      const edit = new WorkspaceEdit();

      const oldLinesCount = oldLines.length;
      const newLinesCount = newLines.length;
      const startDeleteLine = startLine;
      const endDeleteLine = startLine + (newLinesCount - oldLinesCount);
      const deleteRange = new Range(new Position(startDeleteLine, 0), new Position(endDeleteLine, 0));
      edit.delete(Uri.parse(documentUri), deleteRange);
      await workspace.applyEdit(edit);
    }
    this.pendingChanges.delete(documentUri);
    this.pendingCodeLenses.delete(documentUri);
    if (editor && editor.document.uri.toString() === documentUri) {
      editor.setDecorations(this.addedDecorationType, []);
      editor.setDecorations(this.removedDecorationType, []);
    }
    this.fire();
  }

  private async clearAllPendingChanges() {
    for (const documentUri of this.pendingChanges.keys()) {
      await this.clearPendingChanges(documentUri, false);
    }
  }

  async resolveOutline(action: "accept" | "discard") {
    const editor = window.activeTextEditor;
    if (!editor) {
      window.showInformationMessage("No active editor.");
      return;
    }

    const documentUri = editor.document.uri;
    const pendingChange = this.pendingChanges.get(documentUri.toString());

    if (!pendingChange) {
      window.showInformationMessage("No pending changes to resolve.");
      return;
    }

    const { newContent, originalStartLine } = pendingChange;

    if (action === "accept") {
      await this.acceptChanges(documentUri, newContent, originalStartLine);
      window.showInformationMessage("Changes accepted.");
    } else if (action === "discard") {
      await this.discardChanges(documentUri);
      window.showInformationMessage("Changes discarded.");
    }
  }
}
