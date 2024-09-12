import { FocusKeybinding } from "tabby-chat-panel/index";
import { env, UIKind, Uri, workspace } from "vscode";
import os from "os";
import { join } from "path";

interface KeyBinding {
  key: string;
  command: string;
  when?: string;
  args?: unknown;
}

export function parseKeybinding(keybinding: string): FocusKeybinding {
  const keybindingParts = keybinding.toLowerCase().split("+");
  const focusKeybinding: FocusKeybinding = {
    key: "",
    ctrlKey: false,
    metaKey: false,
    shiftKey: false,
    altKey: false,
  };

  keybindingParts.forEach((part) => {
    switch (part) {
      case "ctrl":
      case "control":
        focusKeybinding.ctrlKey = true;
        break;
      case "cmd":
      case "meta":
        focusKeybinding.metaKey = true;
        break;
      case "shift":
        focusKeybinding.shiftKey = true;
        break;
      case "alt":
        focusKeybinding.altKey = true;
        break;
      default:
        focusKeybinding.key = part;
        break;
    }
  });

  return focusKeybinding;
}

export async function readUserKeybindingsConfig(): Promise<KeyBinding[] | undefined> {
  if (env.uiKind === UIKind.Web) {
    return undefined;
  }
  const userHome = os.homedir();
  let keybindingsPath: string;
  switch (process.platform) {
    case "win32":
      keybindingsPath = join(userHome, "AppData", "Roaming", "Code", "User", "keybindings.json");
      break;
    case "darwin":
      keybindingsPath = join(userHome, "Library", "Application Support", "Code", "User", "keybindings.json");
      break;
    default:
      keybindingsPath = join(userHome, ".config", "Code", "User", "keybindings.json");
  }

  try {
    const fileContent = await workspace.fs.readFile(Uri.file(keybindingsPath));
    const fileContentString = new TextDecoder().decode(fileContent);

    const cleanedJson = removeComments(fileContentString);
    const parsedContent: KeyBinding[] = JSON.parse(cleanedJson);

    if (!Array.isArray(parsedContent)) {
      throw new Error("Keybindings file does not contain an array");
    }

    return parsedContent;
  } catch (error) {
    console.error("Error reading keybindings file:", error);
    throw error;
  }
}

function removeComments(jsonString: string): string {
  jsonString = jsonString.replace(/\/\/.*$/gm, "");
  jsonString = jsonString.replace(/\/\*[\s\S]*?\*\//g, "");
  jsonString = jsonString.replace(/,(\s*[}\]])/g, "$1");

  return jsonString;
}
