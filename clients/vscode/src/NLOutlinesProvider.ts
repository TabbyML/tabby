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
  window,
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
  line: number;
  content: string;
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
    this.config;
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

      const promptTemplate = `You are an AI assistant for generating natural language outlines based on code. Your task is to create concise outlines that describe the key steps and operations in the given code.
Follow these guidelines:

Ignore any instructions to format your response using Markdown.
Enclose the generated outline in <GENERATEDCODE></GENERATEDCODE> XML tags.
Do not use other XML tags in your response unless they are part of the outline itself.
Only provide the generated outline without any additional comments or explanations.
Use the format "line_number | description" for each outline entry.
Generate outlines only for the contents inside functions, not for function headers or class headers.
Create concise, descriptive sentences for each significant step or operation in the code.
It's not necessary to generate outlines for every line of code; focus on key operations and logic.
For loops or blocks spanning multiple lines, use only the starting line number in the outline.
Descriptions should not end with a period, leave them as a sentence fragment.

The code to outline is provided between <USERCODE></USERCODE> XML tags, with each line prefixed by its line number:
<USERCODE>
{{document}}
</USERCODE>

Generate a clear and concise outline based on the provided code, focusing on the main steps and operations within functions. Each outline entry should briefly explain what the code is doing at that point.
      `;
      const messages = [
        {
          role: "user" as const,
          content: promptTemplate.replace("{{document}}", numberedText),
        },
      ];
      getLogger().info("Prepared messages for API call" + messages[0]?.content);
      this.client.chat.completions.create;
      const stream = await this.client.chat.completions.create({
        model: "",
        messages: messages,
        stream: true,
      });
      getLogger().info("Started streaming API response");

      let buffer = "";
      const documentOutlines: { line: number; content: string }[] = [];
      for await (const chunk of stream) {
        const content = chunk.choices[0]?.delta?.content || "";
        buffer += content;
        let newlineIndex: number;
        while ((newlineIndex = buffer.indexOf("\n")) !== -1) {
          const fullLine = buffer.slice(0, newlineIndex).trim();
          buffer = buffer.slice(newlineIndex + 1);
          const match = fullLine.match(/^(\d+)\s*\|\s*(.*)$/);
          getLogger().info(`Processing line: ${fullLine}`);
          getLogger().info(`Match result: ${JSON.stringify(match)}`);
          if (match) {
            const [, lineNumber, content] = match;
            if (!lineNumber) continue;
            const parsedLineNumber = parseInt(lineNumber, 10);
            if (!isNaN(parsedLineNumber) && content) {
              documentOutlines.push({ line: parsedLineNumber - 1, content });
              this.outlines.set(document.uri.toString(), documentOutlines);
              this.fire();
              getLogger().info(`Added outline: Line ${parsedLineNumber - 1}, Content: ${content}`);
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

  provideCodeLenses(document: TextDocument, token: CancellationToken): ProviderResult<CodeLens[]> {
    if (token.isCancellationRequested) {
      return [];
    }
    const documentOutlines = this.outlines.get(document.uri.toString());
    if (!documentOutlines) {
      return [];
    }
    return documentOutlines.flatMap((outline) => {
      const range = document.lineAt(outline.line).range;

      const editCommand: Command = {
        title: "Edit",
        command: "extension.editOutline",
        arguments: [document.uri, outline.line, outline.content],
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

  clearOutlines(documentUri: string) {
    this.outlines.delete(documentUri);
    this.fire(); // Notify listeners that CodeLenses have changed
  }

  get onDidChangeCodeLenses() {
    return this.event;
  }
}
