import { extensions, workspace, Uri } from "vscode";
import type { Repository as GitRepository, API } from "./git";
export type Repository = GitRepository;
import { getLogger } from "../logger";

function getGitExtensionApi(): API | undefined {
  const ext = extensions.getExtension("vscode.git");
  return ext?.isActive ? ext.exports.getAPI(1) : undefined;
}

export class GitProvider {
  private readonly logger = getLogger();
  private api: API | undefined = undefined;

  constructor() {
    this.init();
  }

  private init(tries = 0) {
    this.api = getGitExtensionApi();
    if (this.api) {
      this.logger.info("GitProvider created.");
    } else {
      if (tries > 10) {
        this.logger.warn(`Failed to create GitProvider after ${tries} tries, giving up.`);
      } else {
        const delay = (tries + 1) * 1000;
        this.logger.info(`Failed to create GitProvider, retry after ${delay}ms`);
        setTimeout(() => {
          this.init(tries + 1);
        }, delay);
      }
    }
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
