import { EventEmitter } from "events";
import { CancellationToken } from "vscode";
import { BaseLanguageClient, DynamicFeature, FeatureState, RegistrationData } from "vscode-languageclient";
import {
  ServerCapabilities,
  ChatFeatureRegistration,
  GenerateCommitMessageRequest,
  GenerateCommitMessageParams,
  GenerateCommitMessageResult,
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
}
