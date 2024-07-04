import { Uri } from "vscode";
import { BaseLanguageClient, StaticFeature, FeatureState, Disposable } from "vscode-languageclient";
import {
  ClientCapabilities,
  GitRepositoryRequest,
  GitRepositoryParams,
  GitDiffRequest,
  GitDiffParams,
} from "tabby-agent";
import { GitProvider } from "../git/GitProvider";

export class GitProviderFeature implements StaticFeature {
  private disposables: Disposable[] = [];

  constructor(
    private readonly client: BaseLanguageClient,
    private readonly gitProvider: GitProvider,
  ) {}

  getState(): FeatureState {
    return { kind: "static" };
  }

  fillInitializeParams() {
    // nothing
  }

  fillClientCapabilities(capabilities: ClientCapabilities): void {
    capabilities.tabby = {
      ...capabilities.tabby,
      gitProvider: this.gitProvider.isApiAvailable(),
    };
  }

  preInitialize(): void {
    // nothing
  }

  initialize(): void {
    this.disposables.push(
      this.client.onRequest(GitRepositoryRequest.type, (params: GitRepositoryParams) => {
        const repository = this.gitProvider.getRepository(Uri.parse(params.uri));
        if (!repository) {
          return null;
        }
        return {
          root: repository.rootUri.toString(),
          remoteUrl: this.gitProvider.getDefaultRemoteUrl(repository),
          remotes: repository.state.remotes
            .map((remote) => ({
              name: remote.name,
              url: remote.fetchUrl ?? remote.pushUrl ?? "",
            }))
            .filter((remote) => {
              return remote.url.length > 0;
            }),
        };
      }),
    );
    this.disposables.push(
      this.client.onRequest(GitDiffRequest.type, async (params: GitDiffParams) => {
        const repository = this.gitProvider.getRepository(Uri.parse(params.repository));
        if (!repository) {
          return null;
        }
        const diff = await this.gitProvider.getDiff(repository, params.cached);
        if (!diff) {
          return null;
        }
        return { diff };
      }),
    );
  }

  clear(): void {
    this.disposables.forEach((disposable) => disposable.dispose());
    this.disposables = [];
  }
}
