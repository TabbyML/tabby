import { BaseLanguageClient, DynamicFeature, FeatureState, RegistrationData } from "vscode-languageclient";
import { InlineNESCompletionRequest } from "tabby-agent";
import { getLogger } from "../logger";

/**
 * Implementation of the dynamic feature for Next Edit Suggestion (NES) completions
 */
export class InlineNESCompletionFeature implements DynamicFeature<unknown> {
  private readonly logger = getLogger("InlineNESCompletionFeature");
  private registration: string | undefined = undefined;

  constructor(private readonly client: BaseLanguageClient) {}

  // Required properties
  readonly registrationType = InlineNESCompletionRequest.type;

  getState(): FeatureState {
    return { kind: "workspace", id: this.registrationType.method, registrations: this.isAvailable };
  }

  get isAvailable(): boolean {
    return !!this.registration;
  }

  fillClientCapabilities() {
    // No special client capabilities needed
  }

  initialize() {
    // Register request handler
    this.client.onRequest(InlineNESCompletionRequest.type, (params, token) => {
      this.logger.debug("Handling InlineNESCompletionRequest via feature handler");
      return this.client.sendRequest(InlineNESCompletionRequest.method, params, token);
    });
  }

  register(data: RegistrationData<unknown>): void {
    this.registration = data.id;
    this.logger.debug(`Registered InlineNESCompletionRequest with id: ${this.registration}`);
  }

  unregister(id: string): void {
    if (this.registration === id) {
      this.registration = undefined;
      this.logger.debug(`Unregistered InlineNESCompletionRequest with id: ${id}`);
    }
  }

  clear(): void {
    this.registration = undefined;
  }
}
