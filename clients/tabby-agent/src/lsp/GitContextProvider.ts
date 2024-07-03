import { spawn } from "child_process";
import { parse as uriParse, serialize as uriSerialize } from "uri-js";
import { CancellationToken } from "vscode-languageserver-protocol";
import { GitRepositoryParams, GitRepository, GitDiffParams, GitDiffResult } from "./protocol";
import { isBrowser } from "../env";
import { getLogger } from "../logger";
import "../ArrayExt";

export interface GitContextProvider {
  getRepository(params: GitRepositoryParams, token?: CancellationToken): Promise<GitRepository | null>;
  diff(params: GitDiffParams, token?: CancellationToken): Promise<GitDiffResult | null>;
}

const logger = getLogger("GitContextProvider");

async function executeGitCommand(cwd?: string, args: string[] = [], token?: CancellationToken): Promise<string> {
  return new Promise((resolve, reject) => {
    const git = spawn("git", args, {
      cwd,
    });
    let result = "";

    git.stdout.on("data", (data) => {
      result += data.toString();
    });

    git.on("close", (code) => {
      if (code === 0) {
        resolve(result);
      } else {
        reject(`Git command failed, code: ${code}, args: ${args.join(" ")}`);
      }
    });

    if (token?.isCancellationRequested) {
      reject("The request is canceled.");
    }
    token?.onCancellationRequested(() => {
      reject("The request is canceled.");
    });
  });
}

async function isGitCommandAvailable(): Promise<boolean> {
  try {
    await executeGitCommand(undefined, ["--version"]);
    return true;
  } catch (e) {
    logger.debug(`Git command is not available. ${e}`);
    return false;
  }
}

async function getRepository(params: GitRepositoryParams, token?: CancellationToken): Promise<GitRepository | null> {
  try {
    const uri = uriParse(params.uri);
    if (uri.scheme !== "file") {
      return null;
    }
    const root = await executeGitCommand(uri.path, ["rev-parse", "--show-toplevel"], token);
    uri.path = root;
    const remoteVerbose = await executeGitCommand(root, ["remote", "-v"], token);
    const remotes = remoteVerbose
      .split("\n")
      .map((remoteLine) => {
        const [name, url] = remoteLine.trim().split(/\s+/);
        return { name, url };
      })
      .filter<{ name: string; url: string }>((remote): remote is { name: string; url: string } => {
        return !!remote.name && !!remote.url;
      })
      .distinct((item) => item.name);
    return { root: uriSerialize(uri), remotes };
  } catch (e) {
    logger.debug(`Failed to get repository for ${params}. ${e}`);
    return null;
  }
}

async function diff(params: GitDiffParams, token?: CancellationToken): Promise<GitDiffResult | null> {
  try {
    const { repository, cached } = params;
    const uri = uriParse(repository);
    if (uri.scheme !== "file") {
      return null;
    }
    const args = ["diff"];
    if (cached) {
      args.push("--cached");
    }
    const diff = await executeGitCommand(uri.path, args, token);
    return { diff };
  } catch (e) {
    logger.debug(`Failed to get diff for ${params}. ${e}`);
    return null;
  }
}

var gitContextProvider: GitContextProvider | null | undefined = undefined;

export async function getGitContextProvider(): Promise<GitContextProvider | null> {
  if (isBrowser) {
    return null;
  }
  if (gitContextProvider == undefined) {
    if (await isGitCommandAvailable()) {
      gitContextProvider = {
        getRepository,
        diff,
      };
    } else {
      gitContextProvider = null;
    }
  }
  return gitContextProvider;
}
