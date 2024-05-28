import { extensions, workspace, Extension, Uri } from "vscode";
import type { GitExtension, Repository as GitRepository, API } from "./git";
export type Repository = GitRepository;

export class GitProvider {
  private ext: Extension<GitExtension> | undefined;
  private api: API | undefined;
  constructor() {
    this.ext = extensions.getExtension("vscode.git");
    this.api = this.ext?.isActive ? this.ext.exports.getAPI(1) : undefined;
  }

  getRepositories(): Repository[] | undefined {
    return this.api?.repositories;
  }

  getRepository(uri: Uri): Repository | undefined {
    return this.api?.getRepository(uri) ?? undefined;
  }

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
    return remote?.fetchUrl ?? remote?.pushUrl;
  }
}
