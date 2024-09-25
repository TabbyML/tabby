import { spawn } from "child_process";
import * as path from "path";
import * as fs from "fs-extra";
import { parse as uriParse, serialize as uriSerialize } from "uri-js";
import { CancellationToken } from "vscode-languageserver-protocol";
import { GitRepositoryParams, GitRepository, GitDiffParams, GitDiffResult } from "../protocol";
import { isBrowser } from "../env";
import { getLogger } from "../logger";
import "../utils/array";

export interface GitCommandRunner {
  getRepository(params: GitRepositoryParams, token?: CancellationToken): Promise<GitRepository | null>;
  diff(params: GitDiffParams, token?: CancellationToken): Promise<GitDiffResult | null>;
}

const logger = getLogger("GitCommandRunner");

async function executeGitCommand(cwd?: string, args: string[] = [], token?: CancellationToken): Promise<string> {
  return new Promise((resolve, reject) => {
    const git = spawn("git", args, {
      cwd,
    });
    let result = "";

    git.stdout.on("data", (data) => {
      result += data.toString();
    });

    git.on("error", (error) => {
      reject(`Git command error: ${error}, cwd: ${cwd}, args: ${args.join(" ")}`);
    });

    const exitHandler = (code: number | null) => {
      if (code === 0) {
        resolve(result.trim());
      } else {
        reject(`Git command failed, code: ${code}, cwd: ${cwd}, args: ${args.join(" ")}`);
      }
    };
    git.on("exit", exitHandler);
    git.on("close", exitHandler);

    if (token?.isCancellationRequested) {
      reject("The request is canceled.");
    }
    token?.onCancellationRequested(() => {
      reject("The request is canceled.");
    });
  });
}

async function ensureCwd(filepath: string): Promise<string> {
  const stats = await fs.stat(filepath);
  if (stats.isDirectory()) {
    return filepath;
  }
  return path.dirname(filepath);
}

function replaceUriPath(uri: string, path: string): string {
  const uriComponents = uriParse(uri);
  uriComponents.path = path;
  return uriSerialize(uriComponents);
}

async function isGitCommandAvailable(): Promise<boolean> {
  try {
    const version = await executeGitCommand(undefined, ["--version"]);
    logger.debug(`Git command is available, ${version}.`);
    return true;
  } catch (e) {
    logger.debug(`Git command is not available. ${e}`);
    return false;
  }
}

async function getRepository(params: GitRepositoryParams, token?: CancellationToken): Promise<GitRepository | null> {
  try {
    logger.trace("Get repository: ", { params });
    const { scheme, path: filepath } = uriParse(params.uri);
    if (scheme !== "file" || !filepath) {
      return null;
    }
    const cwd = await ensureCwd(filepath);
    const rootPath = await executeGitCommand(cwd, ["rev-parse", "--show-toplevel"], token);
    const root = replaceUriPath(params.uri, rootPath);
    const remoteOutput = await executeGitCommand(rootPath, ["remote", "-v"], token);
    const remotes = remoteOutput
      .split("\n")
      .map((remoteLine) => {
        const [name, url] = remoteLine.trim().split(/\s+/);
        return { name, url };
      })
      .filter<{ name: string; url: string }>((remote): remote is { name: string; url: string } => {
        return !!remote.name && !!remote.url;
      })
      .distinct((item) => item.name);
    const result = { root, remotes };
    logger.trace("Get repository result: ", { result });
    return result;
  } catch (e) {
    logger.debug(`Failed to get repository for ${params.uri}. ${e}`);
    return null;
  }
}

async function diff(params: GitDiffParams, token?: CancellationToken): Promise<GitDiffResult | null> {
  try {
    logger.trace("Get diff: ", { params });
    const { repository, cached } = params;
    const { scheme, path: rootPath } = uriParse(repository);
    if (scheme !== "file" || !rootPath) {
      return null;
    }
    const args = ["diff"];
    if (cached) {
      args.push("--cached");
    }
    const diff = await executeGitCommand(rootPath, args, token);
    const result = { diff };
    logger.trace("Get diff result: ", { result });
    return result;
  } catch (e) {
    logger.debug(`Failed to get diff for ${params.repository}. ${e}`);
    return null;
  }
}

let gitCommandRunner: GitCommandRunner | undefined = undefined;

export async function getGitCommandRunner(): Promise<GitCommandRunner | undefined> {
  if (isBrowser) {
    return undefined;
  }
  if (gitCommandRunner == undefined) {
    if (await isGitCommandAvailable()) {
      gitCommandRunner = {
        getRepository,
        diff,
      };
    } else {
      gitCommandRunner = undefined;
    }
  }
  return gitCommandRunner;
}
