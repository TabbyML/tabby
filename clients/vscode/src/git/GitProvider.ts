import { extensions, workspace, Uri } from "vscode";
import type { Repository as GitRepository, API } from "./git";
export type Repository = GitRepository;
import { getLogger } from "../logger";

export class GitProvider {
  private readonly logger = getLogger();
  private api: API | undefined = undefined;
  private remoteUrlToLocalRoot = new Map<string, Uri | undefined>();

  private async initGitExtensionApi(tries = 0): Promise<void> {
    try {
      const ext = extensions.getExtension("vscode.git");
      if (ext?.isActive) {
        this.api = ext.exports.getAPI(1);
      }
    } catch (err) {
      this.logger.debug(`${err}`);
    }
    if (this.api) {
      this.logger.info("GitProvider created.");
    } else {
      if (tries >= 2) {
        this.logger.warn(`Failed to create GitProvider after ${tries} tries, giving up.`);
      } else {
        const delay = (tries + 1) * 1000;
        this.logger.info(`Failed to create GitProvider, retry after ${delay}ms`);
        await new Promise((resolve) => setTimeout(resolve, delay));
        await this.initGitExtensionApi(tries + 1);
      }
    }
  }

  async init(): Promise<void> {
    await this.initGitExtensionApi();
  }

  isApiAvailable(): boolean {
    return !!this.api;
  }

  getRepositories(): Repository[] | undefined {
    return this.api?.repositories;
  }

  getRepository(uri: Uri): Repository | undefined {
    return this.api?.getRepository(uri) ?? undefined;
  }

  /**
   * Retrieves diff information from the specified git repository.
   * @param repository The Git repository to get the diff from.
   * @param cached When true, shows diff between index and HEAD (staged changes).
   *               When false, shows diff between working tree and index (unstaged changes).
   * @returns Promise resolving to an array of diff strings, each representing a single file's changes,
   *          or undefined if the operation fails. Results are sorted by file modification time.
   */
  async getDiff(repository: Repository, cached: boolean): Promise<string[] | undefined> {
    const diff = (await repository.diff(cached)).trim();
    const diffs = await Promise.all(
      diff.split(/\n(?=diff)/).map(async (item: string) => {
        let priority = Number.MAX_SAFE_INTEGER;
        const filepath = /diff --git a\/.* b\/(.*)$/gm.exec(item)?.[1];
        if (filepath) {
          const uri = Uri.joinPath(repository.rootUri, filepath);
          try {
            priority = (await workspace.fs.stat(uri)).mtime;
          } catch (error) {
            //ignore
          }
        }
        return { diff: item, priority };
      }),
    );
    return diffs.sort((a, b) => a.priority - b.priority).map((item) => item.diff);
  }

  getDefaultRemoteUrl(repository: Repository): string | undefined {
    const remote =
      repository.state.remotes.find((remote) => remote.name === "origin") ||
      repository.state.remotes.find((remote) => remote.name === "upstream") ||
      repository.state.remotes[0];
    const remoteUrl = remote?.fetchUrl ?? remote?.pushUrl;
    if (remoteUrl) {
      this.remoteUrlToLocalRoot.set(remoteUrl, repository.rootUri);
    }
    return remoteUrl;
  }

  findLocalRootUriByRemoteUrl(remoteUrl: string): Uri | undefined {
    if (this.remoteUrlToLocalRoot.has(remoteUrl)) {
      return this.remoteUrlToLocalRoot.get(remoteUrl);
    }
    const allRepos = this.getRepositories();
    const repo = allRepos?.find((repo) =>
      repo.state.remotes.find((remote) => remote.fetchUrl === remoteUrl || remote.pushUrl === remoteUrl),
    );
    const localRootUri = repo?.rootUri;
    this.remoteUrlToLocalRoot.set(remoteUrl, localRootUri);
    return localRootUri;
  }
}
