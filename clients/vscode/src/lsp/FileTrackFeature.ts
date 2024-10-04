import { OpenedFileParams, OpenedFileRequest } from "tabby-agent";
import { getLogger } from "../logger";
import { Client } from "./Client";
import { ExtensionContext, TextEditor, window } from "vscode";
import {
  DocumentSelector,
  FeatureState,
  InitializeParams,
  ServerCapabilities,
  StaticFeature,
} from "vscode-languageclient";
import EventEmitter from "events";
import { FileTrackerProvider } from "../FileTrackProvider";

export class FileTrackerFeature extends EventEmitter implements StaticFeature {
  private readonly fileTrackProvider: FileTrackerProvider = new FileTrackerProvider();
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
        getLogger().info("onDidChangeActiveTextEditor happend:");
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
      const params: OpenedFileParams = {
        action: "change",
        activeEditor: {
          uri: editor.document.uri.toString(),
          visibleRange: {
            start: { line: editorRange.start.line, character: editorRange.start.character },
            end: { line: editorRange.end.line, character: editorRange.end.character },
          },
        },
        visibleEditors: this.fileTrackProvider.collectVisibleEditors(true, editor),
      };
      await this.client.languageClient.sendNotification(OpenedFileRequest.method, params);
    }
  }
}
