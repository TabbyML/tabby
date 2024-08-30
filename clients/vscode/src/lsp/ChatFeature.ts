import { EventEmitter } from "events";
import {
  window,
  workspace,
  Range,
  Position,
  Disposable,
  CancellationToken,
  TextEditorEdit,
  TextDocument,
} from "vscode";
import { BaseLanguageClient, DynamicFeature, FeatureState, RegistrationData, TextEdit } from "vscode-languageclient";
import {
  ServerCapabilities,
  ChatFeatureRegistration,
  GenerateCommitMessageRequest,
  GenerateCommitMessageParams,
  GenerateCommitMessageResult,
  ChatEditCommandRequest,
  ChatEditCommandParams,
  ChatEditCommand,
  ChatEditRequest,
  ChatEditParams,
  ChatEditToken,
  ChatEditResolveRequest,
  ChatEditResolveParams,
  ApplyWorkspaceEditParams,
  ApplyWorkspaceEditRequest,
} from "tabby-agent";
import { diffLines } from "diff";

export class ChatFeature extends EventEmitter implements DynamicFeature<unknown> {
  private registration: string | undefined = undefined;
  private disposables: Disposable[] = [];

  constructor(private readonly client: BaseLanguageClient) {
    super();
  }

  readonly registrationType = ChatFeatureRegistration.type;

  getState(): FeatureState {
    return { kind: "workspace", id: this.registrationType.method, registrations: this.isAvailable };
  }

  fillInitializeParams() {
    // nothing
  }

  fillClientCapabilities(): void {
    // nothing
  }

  preInitialize(): void {
    // nothing
  }

  initialize(capabilities: ServerCapabilities): void {
    if (capabilities.tabby?.chat) {
      this.register({ id: this.registrationType.method, registerOptions: {} });
    }

    this.disposables.push(
      this.client.onRequest(ApplyWorkspaceEditRequest.type, (params: ApplyWorkspaceEditParams) => {
        return this.handleApplyWorkspaceEdit(params);
      }),
    );
  }

  register(data: RegistrationData<unknown>): void {
    this.registration = data.id;
    this.emit("didChangeAvailability", true);
  }

  unregister(id: string): void {
    if (this.registration === id) {
      this.registration = undefined;
      this.emit("didChangeAvailability", false);
    }
  }

  clear(): void {
    this.disposables.forEach((disposable) => disposable.dispose());
    this.disposables = [];
  }

  get isAvailable(): boolean {
    return !!this.registration;
  }

  async generateCommitMessage(
    params: GenerateCommitMessageParams,
    token?: CancellationToken,
  ): Promise<GenerateCommitMessageResult | null> {
    if (!this.isAvailable) {
      return null;
    }
    return this.client.sendRequest(GenerateCommitMessageRequest.method, params, token);
  }

  // target is where the fetched command will be filled in
  // callback will be called when target updated
  async provideEditCommands(
    params: ChatEditCommandParams,
    target: {
      commands: ChatEditCommand[];
      callback: () => void;
    },
    token?: CancellationToken,
  ): Promise<void> {
    // FIXME: handle partial results after server supports partial results
    const commands: ChatEditCommand[] | null = await this.client.sendRequest(
      ChatEditCommandRequest.method,
      params,
      token,
    );
    if (commands && commands.length > 0) {
      target.commands.push(...commands);
      target.callback();
    }
  }

  async provideEdit(params: ChatEditParams, token?: CancellationToken): Promise<ChatEditToken | null> {
    if (!this.isAvailable) {
      return null;
    }
    return this.client.sendRequest(ChatEditRequest.method, params, token);
  }

  private async handleApplyWorkspaceEdit(params: ApplyWorkspaceEditParams): Promise<boolean> {
    const { edit, options } = params;
    const activeEditor = window.activeTextEditor;
    if (!activeEditor) {
      return false;
    }

    try {
      const success = await activeEditor.edit(
        (editBuilder: TextEditorEdit) => {
          Object.entries(edit.changes || {}).forEach(([uri, textEdits]) => {
            const document = workspace.textDocuments.find((doc) => doc.uri.toString() === uri);
            if (document && document === activeEditor.document) {
              const textEdit = textEdits[0];
              if (textEdits.length === 1 && textEdit) {
                applyTextEditMinimalLineChange(editBuilder, textEdit, document);
              } else {
                textEdits.forEach((textEdit) => {
                  const range = new Range(
                    new Position(textEdit.range.start.line, textEdit.range.start.character),
                    new Position(textEdit.range.end.line, textEdit.range.end.character),
                  );
                  editBuilder.replace(range, textEdit.newText);
                });
              }
            }
          });
        },
        {
          undoStopBefore: options?.undoStopBefore ?? false,
          undoStopAfter: options?.undoStopAfter ?? false,
        },
      );

      return success;
    } catch (error) {
      return false;
    }
  }

  async resolveEdit(params: ChatEditResolveParams): Promise<boolean> {
    return this.client.sendRequest(ChatEditResolveRequest.method, params);
  }
}

function applyTextEditMinimalLineChange(editBuilder: TextEditorEdit, textEdit: TextEdit, document: TextDocument) {
  const documentRange = new Range(
    new Position(textEdit.range.start.line, textEdit.range.start.character),
    new Position(textEdit.range.end.line, textEdit.range.end.character),
  );

  const text = document.getText(documentRange);
  const newText = textEdit.newText;
  const diffs = diffLines(text, newText);

  let line = documentRange.start.line;
  for (const diff of diffs) {
    if (!diff.count) {
      continue;
    }

    if (diff.added) {
      editBuilder.insert(new Position(line, 0), diff.value);
    } else if (diff.removed) {
      const range = new Range(new Position(line + 0, 0), new Position(line + diff.count, 0));
      editBuilder.delete(range);
      line += diff.count;
    } else {
      line += diff.count;
    }
  }
}
