import { FocusKeybinding } from "tabby-chat-panel/index";

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
