import { DidChangeActiveEditorNotification, DidChangeActiveEditorParams } from "tabby-agent";
import { Client } from "./client";
import { ExtensionContext, TextEditor, window } from "vscode";
import {
  DocumentSelector,
  FeatureState,
  InitializeParams,
  ServerCapabilities,
  StaticFeature,
} from "vscode-languageclient";
import EventEmitter from "events";
import { collectVisibleEditors } from "../windowUtils";

export class FileTrackerFeature extends EventEmitter implements StaticFeature {
  constructor(
    private readonly client: Client,
    private readonly context: ExtensionContext,
  ) {
    super();
  }
  fillInitializeParams?: ((params: InitializeParams) => void) | undefined;
  fillClientCapabilities(): void {
    //nothing
  }
  preInitialize?:
    | ((capabilities: ServerCapabilities, documentSelector: DocumentSelector | undefined) => void)
    | undefined;
  initialize(): void {
    this.context.subscriptions.push(
      //when active text editor changes
      window.onDidChangeActiveTextEditor(async (editor) => {
        await this.addingChangeEditor(editor);
      }),
    );
  }
  getState(): FeatureState {
    throw new Error("Method not implemented.");
  }
  clear(): void {
    throw new Error("Method not implemented.");
  }

  async addingChangeEditor(editor: TextEditor | undefined) {
    if (editor && editor.visibleRanges[0] && editor.document.fileName.startsWith("/")) {
      const editorRange = editor.visibleRanges[0];
      const params: DidChangeActiveEditorParams = {
        activeEditor: {
          uri: editor.document.uri.toString(),
          range: {
            start: { line: editorRange.start.line, character: editorRange.start.character },
            end: { line: editorRange.end.line, character: editorRange.end.character },
          },
        },
        visibleEditors: collectVisibleEditors(true, editor),
      };
      await this.client.languageClient.sendNotification(DidChangeActiveEditorNotification.method, params);
    }
  }
}
