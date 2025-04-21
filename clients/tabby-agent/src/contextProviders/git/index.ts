import type { Connection, CancellationToken } from "vscode-languageserver";
import type { Feature } from "../../feature";
import type { GitCommandRunner } from "./gitCommand";
import {
  ClientCapabilities,
  ServerCapabilities,
  GitRepositoryParams,
  GitRepository,
  GitRepositoryRequest,
  GitDiffParams,
  GitDiffResult,
  GitDiffRequest,
} from "../../protocol";
import { getGitCommandRunner } from "./gitCommand";

export interface GitContext {
  repository: GitRepository;
}

export class GitContextProvider implements Feature {
  private lspConnection: Connection | undefined = undefined;
  private gitCommandRunner: GitCommandRunner | undefined = undefined;

  async initialize(connection: Connection, clientCapabilities: ClientCapabilities): Promise<ServerCapabilities> {
    if (clientCapabilities.tabby?.gitProvider) {
      this.lspConnection = connection;
    } else {
      this.gitCommandRunner = await getGitCommandRunner();
    }
    return {};
  }

  async getContext(uri: string, token: CancellationToken): Promise<GitContext | null> {
    const repository = await this.getRepository({ uri: uri }, token);
    if (repository) {
      return { repository };
    }
    return null;
  }

  async getRepository(params: GitRepositoryParams, token: CancellationToken): Promise<GitRepository | null> {
    if (this.lspConnection) {
      return await this.lspConnection.sendRequest(GitRepositoryRequest.type, params, token);
    } else if (this.gitCommandRunner) {
      return await this.gitCommandRunner.getRepository(params, token);
    }
    return null;
  }

  async diff(params: GitDiffParams, token: CancellationToken): Promise<GitDiffResult | null> {
    if (this.lspConnection) {
      return await this.lspConnection.sendRequest(GitDiffRequest.type, params, token);
    } else if (this.gitCommandRunner) {
      return await this.gitCommandRunner.diff(params, token);
    }
    return null;
  }
}
