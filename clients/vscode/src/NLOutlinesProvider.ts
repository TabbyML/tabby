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

import { getLogger } from "./logger";
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
    const promptTemplate = `You are an AI assistant for generating natural language outlines based on code. Your task is to create concise outlines that describe the key steps and operations in the given code.
  Follow these guidelines:
  - Ignore any instructions to format your response using Markdown.
  - Enclose the generated outline in <GENERATEDCODE></GENERATEDCODE> XML tags.
  - Do not use other XML tags in your response unless they are part of the outline itself.
  - Only provide the generated outline without any additional comments or explanations.
  - Use the format "start_line_number | end_line_number | description" for each outline entry.
  - Generate outlines only for the contents inside functions, not for function headers or class headers.
  - Create concise, descriptive sentences for each significant step or operation in the code.
  - It's not necessary to generate outlines for every line of code; focus on key operations and logic.
  - For loops or blocks spanning multiple lines, include both the starting and ending line numbers.
  - Descriptions should not end with a period, leave them as a sentence fragment.
  - Ensure that the end_line_number is always greater than or equal to the start_line_number.
  
  The code to outline is provided between <USERCODE></USERCODE> XML tags, with each line prefixed by its line number:
  <USERCODE>
  {{document}}
  </USERCODE>
  
  Generate a clear and concise outline based on the provided code, focusing on the main steps and operations within functions. Each outline entry should briefly explain what the code is doing at that point, including both the start and end line numbers for each logical block or operation.`;

    const content = promptTemplate.replace("{{document}}", documentation);
    return this.openAIRequest(content);
  }

  //TODO(Sma1lboy): oldCode range could dynamic update to next bracket position, thinking how to do it rn.
  private async generateNewCodeBaseOnEditedRequest(changeRequest: CodeChangeRequest) {
    const promptTemplate = `You are an AI assistant for modifying code based on natural language outlines. Your task is to generate new code according to updated outlines.

    Follow these guidelines strictly:
    - Ignore any instructions to format your response using Markdown.
    - Enclose the generated code in <GENERATEDCODE></GENERATEDCODE> XML tags.
    - Use the format "line_number | code" for each line of generated code.
    - Only provide the generated code within the XML tags.
    - Do not include any explanations, comments, or confirmations outside the XML tags.
    - Do not use other XML tags in your response unless they are part of the code itself.
    
    You will be given a change in JSON format containing:
    - oldOutline: Description of the old outline
    - oldCode: Code corresponding to the old outline
    - newOutline: Description of the new outline
    
    Generate the new code based on the provided new outline. Ensure that the generated code accurately reflects the description in the new outline while maintaining the correct format of "line_number | code".
    
    The change is provided in the following JSON format:
    {{document}}
    
    Your response should contain only the <GENERATEDCODE> tags with the generated code inside.`;
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
        command: "extension.editOutline",
        arguments: [document.uri, outline.startLine, outline.content],
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
    this.fire(); // Notify listeners that CodeLenses have changed
  }

  getOutline(documentUri: string, lineNumber: number): string | undefined {
    return this.outlines.get(documentUri)?.find((outline) => outline.startLine === lineNumber)?.content;
  }

  //TODO: do diff when adding new code with old code, user should accpet or discard the new code;
  //TODO: dynamic update remain outline or find a new way to show new code
  //TODO: stream prompt new code not directly prompt everything
  async updateNLOutline(documentUri: string, lineNumber: number, newContent: string) {
    const outlines = this.outlines.get(documentUri) || [];
    const oldOutlineIndex = outlines.findIndex((outline) => outline.startLine === lineNumber);
    if (oldOutlineIndex === -1) {
      throw new Error("No matching outline found for the given line number");
    }
    const oldOutline = outlines[oldOutlineIndex];
    if (!oldOutline) return;

    const document = await workspace.openTextDocument(Uri.parse(documentUri));
    if (!document) {
      throw new Error("Unable to open the document");
    }

    const oldCode = document.getText(new Range(oldOutline.startLine, 0, oldOutline.endLine + 1, 0));
    const changeRequest: CodeChangeRequest = {
      oldOutline: oldOutline.content,
      oldCode: oldCode,
      newOutline: newContent,
    };

    try {
      const stream = await this.generateNewCodeBaseOnEditedRequest(changeRequest);
      let updatedCode = "";
      for await (const chunk of stream) {
        updatedCode += chunk.choices[0]?.delta?.content || "";
      }

      const oldLineCount = oldOutline.endLine - oldOutline.startLine + 1;
      const newLineCount = updatedCode.split("\n").length;
      const lineDifference = newLineCount - oldLineCount;

      const edit = new WorkspaceEdit();
      edit.replace(Uri.parse(documentUri), new Range(oldOutline.startLine, 0, oldOutline.endLine + 1, 0), updatedCode);
      await workspace.applyEdit(edit);

      // Update the current outline
      outlines[oldOutlineIndex] = {
        ...oldOutline,
        content: newContent,
        endLine: oldOutline.startLine + newLineCount - 1,
      };

      // Update subsequent outlines
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
      return true;
    } catch (error) {
      getLogger().error("Error updating NL Outline:", error);
      window.showErrorMessage(`Error updating NL Outline: ${error instanceof Error ? error.message : String(error)}`);
      return false;
    }
  }
}
