import { EventEmitter } from "events";
import { CancellationToken } from "vscode";
import { BaseLanguageClient, DynamicFeature, FeatureState, RegistrationData } from "vscode-languageclient";
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
} from "tabby-agent";

export class ChatFeature extends EventEmitter implements DynamicFeature<unknown> {
  private registration: string | undefined = undefined;
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
    // nothing
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

  async resolveEdit(params: ChatEditResolveParams): Promise<boolean> {
    return this.client.sendRequest(ChatEditResolveRequest.method, params);
  }
}
