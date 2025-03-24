import { EventEmitter } from "events";
import { Disposable, CancellationToken } from "vscode";
import { BaseLanguageClient, DynamicFeature, FeatureState, RegistrationData } from "vscode-languageclient";
import {
  ChatFeatures,
  GenerateCommitMessageRequest,
  GenerateCommitMessageParams,
  GenerateCommitMessageResult,
  GenerateBranchNameRequest,
  GenerateBranchNameParams,
  GenerateBranchNameResult,
  ChatEditCommandRequest,
  ChatEditCommandParams,
  ChatEditCommand,
  ChatEditRequest,
  ChatEditParams,
  ChatEditToken,
  ChatEditResolveRequest,
  ChatEditResolveParams,
  SmartApplyParams,
  SmartApplyRequest,
} from "tabby-agent";

export class ChatFeature extends EventEmitter implements DynamicFeature<unknown> {
  private registration: string | undefined = undefined;
  private disposables: Disposable[] = [];

  constructor(private readonly client: BaseLanguageClient) {
    super();
  }

  readonly registrationType = ChatFeatures.type;

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

  initialize(): void {
    // nothing
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

  async generateBranchName(
    params: GenerateBranchNameParams,
    token?: CancellationToken,
  ): Promise<GenerateBranchNameResult | null> {
    if (!this.isAvailable) {
      return null;
    }
    return this.client.sendRequest(GenerateBranchNameRequest.method, params, token);
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

  async provideSmartApplyEdit(params: SmartApplyParams, token?: CancellationToken): Promise<boolean | null> {
    if (!this.isAvailable) {
      return null;
    }
    return this.client.sendRequest(SmartApplyRequest.method, params, token);
  }

  async resolveEdit(params: ChatEditResolveParams): Promise<boolean> {
    return this.client.sendRequest(ChatEditResolveRequest.method, params);
  }
}
