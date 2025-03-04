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

export class KeyBindingManager {
  // Singleton instance to manage keybindings.
  private static instance: KeyBindingManager | null = null;

  /**
   * Returns the singleton instance.
   */
  public static getInstance(): KeyBindingManager {
    if (!KeyBindingManager.instance) {
      KeyBindingManager.instance = new KeyBindingManager();
    }
    return KeyBindingManager.instance;
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
  private async readKeyBindings(): Promise<KeyBinding[] | null> {
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
  private async readWorkspaceKeybindings(): Promise<string> {
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
  private parseKeybindings(data: string): KeyBinding[] {
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
   * Retrieves the keybinding for a specified command from the cached keybindings.
   * Returns the key if found and not disabled; otherwise returns null.
   */
  getKeybinding(command: string): string | null {
    if (!this.keybindings) {
      return null;
    }

    const disabledBinding = this.keybindings.find((b) => b.command === `-${command}`);
    if (disabledBinding) {
      return null;
    }

    const customBinding = this.keybindings.find((b) => b.command === command);
    if (customBinding && customBinding.key) {
      return customBinding.key;
    }

    return getPackageCommandBinding(command);
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

export const formatShortcut = (shortcut: string): string => {
  const isMacOS = !isBrowser && process.platform === "darwin";

  const config = isMacOS
    ? {
        ctrlKey: "⌃",
        shiftKey: "⇧",
        altKey: "⌥",
        metaKey: "⌘",
        separator: "+",
      }
    : {
        ctrlKey: "Ctrl",
        shiftKey: "Shift",
        altKey: "Alt",
        metaKey: isBrowser ? "Windows" : process.platform === "win32" ? "Windows" : "Super",
        separator: "+",
      };

  return shortcut
    .split("+")
    .map((key) => {
      const lowerKey = key.toLowerCase();
      switch (lowerKey) {
        case "ctrl":
        case "control":
          return config.ctrlKey;
        case "shift":
          return config.shiftKey;
        case "alt":
        case "option":
          return config.altKey;
        case "cmd":
        case "command":
        case "meta":
          return config.metaKey;
        default:
          if (lowerKey.length === 1) {
            return lowerKey.toUpperCase();
          }
          return key.charAt(0).toUpperCase() + key.slice(1).toLowerCase();
      }
    })
    .join(config.separator);
};
