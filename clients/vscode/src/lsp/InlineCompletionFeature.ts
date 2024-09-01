import { languages, InlineCompletionItemProvider, Disposable } from "vscode";
import { InlineCompletionRegistrationOptions, FeatureClient } from "vscode-languageclient";
import {
  InlineCompletionMiddleware,
  InlineCompletionItemFeature as VscodeLspInlineCompletionItemFeature,
} from "vscode-languageclient/lib/common/inlineCompletion";

export class InlineCompletionFeature extends VscodeLspInlineCompletionItemFeature {
  constructor(
    client: FeatureClient<InlineCompletionMiddleware>,
    private readonly inlineCompletionItemProvider: InlineCompletionItemProvider,
  ) {
    super(client);
  }

  override registerLanguageProvider(
    options: InlineCompletionRegistrationOptions,
  ): [Disposable, InlineCompletionItemProvider] {
    const selector = this._client.protocol2CodeConverter.asDocumentSelector(options.documentSelector ?? ["**"]);
    const provider = this.inlineCompletionItemProvider;
    return [languages.registerInlineCompletionItemProvider(selector, provider), provider];
  }
}
