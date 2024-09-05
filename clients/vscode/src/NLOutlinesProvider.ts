import {
  CancellationToken,
  CodeLens,
  CodeLensProvider,
  Command,
  EventEmitter,
  Location,
  ProviderResult,
  Range,
  TextDocument,
  TextEditor,
  Uri,
  window,
  workspace,
  WorkspaceEdit,
} from "vscode";
import { Config } from "./Config";
import OpenAI from "openai";
import generateNLOutlinesPrompt from "../assets/prompts/generateNLOutlines.txt";
import editNLOutline from "../assets/prompts/editNLOutline.txt";
import { getLogger } from "./logger";
import * as Diff from "diff";
interface ChatNLOutlinesParams {
  /**
   * The document location to get the outlines for.
   */
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

export class NLOutlinesProvider extends EventEmitter<void> implements CodeLensProvider {
  private client: OpenAI;
  private outlines: Map<string, Outline[]>;

  constructor(private readonly config: Config) {
    super();
    this.client = new OpenAI({
      apiKey: config.serverToken,
      baseURL: config.serverEndpoint + "/v1",
    });
    this.outlines = new Map();
  }

  async provideNLOutlinesGenerate(params: ChatNLOutlinesParams): Promise<boolean> {
    getLogger().info(this.config.serverEndpoint);
    getLogger().info(this.config.serverToken);
    getLogger().info("Entering provideNLOutlinesGenerate method");

    if (!params.editor) {
      getLogger().info("No editor provided in params");
      return false;
    }

    const document = params.editor.document;
    getLogger().info(`Processing document: ${document.uri}`);

    try {
      const selection = new Range(params.location.range.start, params.location.range.end);
      if (selection.isEmpty) {
        getLogger().warn("Empty selection detected");
        throw new Error("No document selected");
      }

      const selectedText = document.getText(selection);
      getLogger().info(`Selected text length: ${selectedText.length}`);
      if (selectedText.length > 3000) {
        getLogger().warn("Selected text exceeds maximum length");
        throw new Error("Document too long");
      }

      const lines = selectedText.split("\n");
      const startLine = selection.start.line;
      const numberedText = lines.map((line, index) => `${startLine + index + 1} | ${line}`).join("\n");
      getLogger().info("Prepared numbered text for processing");

      const stream = await this.generateNLOutlinesRequest(numberedText);
      getLogger().info("Started streaming API response");

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
          getLogger().info(`Processing line: ${fullLine}`);
          getLogger().info(`Match result: ${JSON.stringify(match)}`);

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
              getLogger().info(`Added outline: Lines ${parsedStartLine - 1}-${parsedEndLine - 1}, Content: ${content}`);
            }
          }
        }
      }

      getLogger().info(`Processed ${documentOutlines.length} outline entries`);
      this.outlines.set(document.uri.toString(), documentOutlines);
      getLogger().info(`Set outlines for document: ${document.uri}`);
      this.fire(); // Notify listeners that CodeLenses have changed
      getLogger().info("Notified listeners of CodeLenses change");

      return true;
    } catch (error) {
      getLogger().error("Error generating outlines:", error);
      console.error("Error generating outlines:", error);
      window.showErrorMessage(`Error generating outlines: ${error instanceof Error ? error.message : String(error)}`);
      return false;
    } finally {
      getLogger().info("Exiting provideNLOutlinesGenerate method");
    }
  }

  private async generateNLOutlinesRequest(
    documentation: string,
  ): Promise<AsyncIterable<OpenAI.Chat.Completions.ChatCompletionChunk>> {
    const promptTemplate = editNLOutline;
    const content = promptTemplate.replace("{{document}}", documentation);
    return this.openAIRequest(content);
  }

  //TODO(Sma1lboy): oldCode range could dynamic update to next bracket position, thinking how to do it rn.
  private async generateNewCodeBaseOnEditedRequest(changeRequest: CodeChangeRequest) {
    const promptTemplate = generateNLOutlinesPrompt;
    const changeJson = JSON.stringify(changeRequest, null, 2);

    const content = promptTemplate.replace("{{document}}", changeJson);
    return this.openAIRequest(content);
  }

  private async openAIRequest(question: string) {
    const messages = [
      {
        role: "user" as const,
        content: question,
      },
    ];

    getLogger().info("Prepared messages for API call" + messages[0]?.content);

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
    const documentOutlines = this.outlines.get(document.uri.toString());
    if (!documentOutlines) {
      return [];
    }
    return documentOutlines.flatMap((outline) => {
      const range = document.lineAt(outline.startLine).range;

      const editCommand: Command = {
        title: "Edit",
        command: "tabby.chat.edit.editNLOutline",
        arguments: [document.uri, outline.startLine],
      };
      const editCodeLens = new CodeLens(range, editCommand);

      const outlineCommand: Command = {
        title: outline.content,
        command: "",
        arguments: [],
      };
      const outlineCodeLens = new CodeLens(range, outlineCommand);

      return [editCodeLens, outlineCodeLens];
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

  //TODO(Sma1lboy): do diff when adding new code with old code, user should accpet or discard the new code;
  //TODO(Sma1lboy): dynamic update remain outline or find a new way to show new code
  //TODO(Sma1lboy): stream prompt new code not directly prompt everything
  async updateNLOutline(documentUri: string, lineNumber: number, newContent: string) {
    getLogger().info(`Starting updateNLOutline for document: ${documentUri}, line: ${lineNumber}`);
    const outlines = this.outlines.get(documentUri) || [];
    const oldOutlineIndex = outlines.findIndex((outline) => outline.startLine === lineNumber);
    if (oldOutlineIndex === -1) {
      getLogger().info(`No matching outline found for line number: ${lineNumber}`);
      throw new Error("No matching outline found for the given line number");
    }
    const oldOutline = outlines[oldOutlineIndex];
    if (!oldOutline) {
      getLogger().info("Old outline is undefined, returning");
      return;
    }
    getLogger().info(`Found old outline: ${JSON.stringify(oldOutline)}`);
    const document = await workspace.openTextDocument(Uri.parse(documentUri));
    if (!document) {
      getLogger().info(`Unable to open document: ${documentUri}`);
      throw new Error("Unable to open the document");
    }
    const oldCodeRange = new Range(oldOutline.startLine, 0, oldOutline.endLine + 1, 0);
    const oldCode = document.getText(oldCodeRange);
    const oldCodeWithLineNumbers = this.formatCodeWithLineNumbers(oldCode, oldOutline.startLine);
    const changeRequest: CodeChangeRequest = {
      oldOutline: oldOutline.content,
      oldCode: oldCodeWithLineNumbers,
      newOutline: newContent,
    };
    try {
      getLogger().info("Generating new code based on edited request");
      const stream = await this.generateNewCodeBaseOnEditedRequest(changeRequest);
      let updatedCode = "";
      for await (const chunk of stream) {
        updatedCode += chunk.choices[0]?.delta?.content || "";
      }
      getLogger().info("Received updated code from AI model:", updatedCode);
      // Remove XML tags while preserving newlines
      updatedCode = updatedCode.replace(/<GENERATEDCODE>\n?/, "").replace(/\n?<\/GENERATEDCODE>/, "");

      // Split the code into lines, but keep the last newline if it exists
      let lines = updatedCode.split(/\n(?!$)/);

      lines = lines.map((line) => {
        if (line.trim() === "") {
          return line;
        }
        const parts = line.split("|");
        if (parts.length > 1) {
          const codeContent = parts.slice(1).join("|");
          const leadingWhitespace = codeContent.substring(0, codeContent.length - codeContent.trimStart().length);
          return leadingWhitespace + codeContent.trim();
        }
        return line;
      });
      lines.pop();
      getLogger().info("asd", lines);

      updatedCode = lines.join("\n");

      updatedCode = updatedCode.replace(/\n*$/, "") + "\n";

      getLogger().info("After removing tags and line numbers:", updatedCode);

      //improve this
      const editId = `tabby-${Math.random().toString(36).substr(2, 6)}`;
      const previewLines = this.generateChangesPreview(oldCode, updatedCode, editId);

      // Apply the changes to the document
      const edit = new WorkspaceEdit();
      edit.replace(Uri.parse(documentUri), oldCodeRange, previewLines.join("\n"));
      await workspace.applyEdit(edit);
      getLogger().info("Applied edit with preview to workspace");

      const newLineCount = previewLines.length;
      const lineDifference = newLineCount - (oldOutline.endLine - oldOutline.startLine + 1);
      outlines[oldOutlineIndex] = {
        ...oldOutline,
        content: newContent,
        endLine: oldOutline.startLine + newLineCount - 1,
      };
      getLogger().info(`Updated outline: ${JSON.stringify(outlines[oldOutlineIndex])}`);
      for (let i = oldOutlineIndex + 1; i < outlines.length; i++) {
        const currentOutline = outlines[i];
        if (currentOutline) {
          outlines[i] = {
            ...currentOutline,
            startLine: currentOutline.startLine + lineDifference,
            endLine: currentOutline.endLine + lineDifference,
          };
        }
      }
      this.outlines.set(documentUri, outlines);
      this.fire();
      getLogger().info("Update completed successfully");
      return true;
    } catch (error) {
      getLogger().error("Error updating NL Outline:", error);
      window.showErrorMessage(`Error updating NL Outline: ${error instanceof Error ? error.message : String(error)}`);
      return false;
    }
  }

  async resolveOutline(action: string) {
    action;
    return;
  }

  private formatCodeWithLineNumbers(code: string, startLine: number): string {
    return code
      .split("\n")
      .map((line, index) => `${startLine + index} | ${line}`)
      .join("\n");
  }

  //TODO: remove this
  private generateChangesPreview(oldCode: string, newCode: string, editId: string): string[] {
    const lines: string[] = [];
    let markers = "";
    const stateDescription = "Editing completed";

    lines.push(`<<<<<<< ${stateDescription} <${markers}>[${editId}]`);
    markers += "<";

    const removeLineNumbers = (code: string) =>
      code
        .split("\n")
        .map((line) => {
          const parts = line.split("|");
          return parts.length > 1 ? parts.slice(1).join("|").trim() : line;
        })
        .join("\n");

    const diffs = Diff.diffLines(removeLineNumbers(oldCode), removeLineNumbers(newCode));
    diffs.forEach((diff) => {
      const diffLines = diff.value.split("\n").filter((line) => line !== "");
      diffLines.forEach((line) => {
        if (diff.added) {
          lines.push(line);
          markers += "+";
        } else if (diff.removed) {
          lines.push(line);
          markers += "-";
        } else {
          lines.push(line);
          markers += "=";
        }
      });
    });

    lines.push(`>>>>>>> ${stateDescription} <${markers}>[${editId}]`);

    if (lines[0]) {
      lines[0] = lines[0].replace("<>", `<${markers}>`);
    }
    return lines;
  }
}
