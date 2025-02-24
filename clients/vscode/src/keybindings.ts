import { readFile } from "fs-extra";
import path from "path";
import { getLogger } from "./logger";
import * as vscode from "vscode";
import { isBrowser } from "./env";
import pkg from "../package.json";
interface KeyBinding {
  key: string;
  command: string;
  when?: string;
}

const logger = getLogger("VSCodeKeyBindingManager");
const isMac = isBrowser
  ? navigator.userAgent.toLowerCase().includes("mac")
  : process.platform.toLowerCase().includes("darwin");

export class VSCodeKeyBindingManager {
  // Singleton instance to manage keybindings.
  private static instance: VSCodeKeyBindingManager | null = null;

  /**
   * Returns the singleton instance.
   */
  public static getInstance(): VSCodeKeyBindingManager {
    if (!VSCodeKeyBindingManager.instance) {
      VSCodeKeyBindingManager.instance = new VSCodeKeyBindingManager();
    }
    return VSCodeKeyBindingManager.instance;
  }

  // Cached keybindings loaded during extension startup.
  private keybindings: KeyBinding[] | null = null;

  /**
   * Initializes the keybinding manager.
   * This method should be called once during extension startup.
   * It reads the keybindings.json file once to avoid additional overhead.
   */
  public async init(): Promise<void> {
    this.keybindings = await this.readKeyBindings();
  }

  /**
   * Reads the keybindings file and returns the parsed keybindings.
   */
  async readKeyBindings(): Promise<KeyBinding[] | null> {
    try {
      let rawData: string;
      if (isBrowser) {
        rawData = await this.readWorkspaceKeybindings();
      } else {
        const isMac = process.platform === "darwin";
        const keybindingsPath = isMac
          ? path.join(process.env["HOME"] ?? "~", "Library", "Application Support", "Code", "User", "keybindings.json")
          : path.join(process.env["APPDATA"] || process.env["HOME"] + "/.config", "Code", "User", "keybindings.json");
        rawData = await readFile(keybindingsPath, "utf8");
      }
      return this.parseKeybindings(rawData);
    } catch (error) {
      logger.error("Error reading keybindings:", error);
      return null;
    }
  }

  /**
   * Reads keybindings.json from the workspace folder (for browser environments).
   */
  async readWorkspaceKeybindings(): Promise<string> {
    const workspace = vscode.workspace.workspaceFolders?.[0];
    if (!workspace) {
      throw new Error("No workspace found");
    }
    const keybindingsUri = vscode.Uri.joinPath(workspace.uri, ".vscode", "keybindings.json");
    const data = await vscode.workspace.fs.readFile(keybindingsUri);
    return Buffer.from(data).toString("utf8");
  }

  /**
   * Parses the raw keybindings JSON data and filters out invalid entries.
   */
  parseKeybindings(data: string): KeyBinding[] {
    const cleanData = data
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => !line.startsWith("//"))
      .join("\n");

    try {
      const parsed = JSON.parse(cleanData) as KeyBinding[];
      return parsed.filter((binding) => {
        const isValid = binding.key && binding.command;
        if (!isValid) {
          logger.warn("Invalid keybinding found:", binding);
        }
        return isValid;
      });
    } catch (error) {
      logger.error("Error parsing keybindings JSON:", error);
      return [];
    }
  }

  /**
   * Checks if the specified command is rebound by verifying its presence in the cached keybindings.
   */
  isCommandRebound(command: string): boolean {
    return this.keybindings ? this.keybindings.some((binding) => binding.command === command) : false;
  }

  /**
   * Retrieves the keybinding for a specified command from the cached keybindings.
   * Returns the key if found and not disabled; otherwise returns null.
   */
  getCommandBinding(command: string): string | null {
    if (!this.keybindings) {
      return null;
    }
    const binding = this.keybindings.find((b) => b.command === command && !b.command.startsWith("-"));
    return binding?.key || null;
  }

  /**
   * Checks if a command is disabled by verifying if a keybinding with a '-' prefix exists.
   */
  isKeyBindingDisabled(command: string): boolean {
    return this.keybindings ? this.keybindings.some((b) => b.command === `-${command}`) : false;
  }
}

export function getPackageCommandBinding(command: string): string {
  try {
    if (!pkg.contributes.keybindings || !Array.isArray(pkg.contributes.keybindings)) {
      logger.warn("No keybindings found in package.json");
      return "";
    }
    const binding = pkg.contributes.keybindings.find((b) => b.command === command);
    if (!binding) {
      return "";
    }
    return isMac && binding.mac ? binding.mac : binding.key;
  } catch (error) {
    logger.error("Error reading package.json keybindings:", error);
    return "";
  }
}
