import { StatusBarAlignment, ThemeColor, window, workspace } from "vscode";
import { TabbyClient } from "./TabbyClient";

const label = "Tabby";
const iconLoading = "$(loading~spin)";
const iconReady = "$(check)";
const iconDisconnected = "$(plug)";
const iconDisabled = "$(x)";
const colorNormal = new ThemeColor('statusBar.foreground');
const colorWarning = new ThemeColor('statusBarItem.warningForeground');
const backgroundColorNormal = new ThemeColor('statusBar.background');
const backgroundColorWarning = new ThemeColor('statusBarItem.warningBackground');

const item = window.createStatusBarItem(StatusBarAlignment.Right);
export const tabbyStatusBarItem = item;

const client = TabbyClient.getInstance();
client.on("statusChanged", updateStatusBarItem);

workspace.onDidChangeConfiguration((event) => {
  if (event.affectsConfiguration("tabby")) {
    updateStatusBarItem();
  }
});

updateStatusBarItem();
item.show();

function updateStatusBarItem() {
  const enabled = workspace.getConfiguration("tabby").get("enabled", true);
  if (!enabled) {
    toDisabled();
  } else {
    const status = client.status;
    switch (status) {
      case "connecting":
        toLoading();
        break;
      case "ready":
        toReady();
        break;
      case "disconnected":
        toDisconnected();
        break;
    }
  }
}

function toLoading() {
  item.color = colorNormal;
  item.backgroundColor = backgroundColorNormal;
  item.text = `${iconLoading} ${label}`;
  item.tooltip = "Connecting to Tabby Server...";
  item.command = undefined;
}

function toReady() {
  item.color = colorNormal;
  item.backgroundColor = backgroundColorNormal;
  item.text = `${iconReady} ${label}`;
  item.tooltip = "Tabby is providing code suggestions for you.";
  item.command = "tabby.toggleEnabled";
}

function toDisconnected() {
  item.color = colorWarning;
  item.backgroundColor = backgroundColorWarning;
  item.text = `${iconDisconnected} ${label}`;
  item.tooltip = "Cannot connect to Tabby Server. Click to open settings.";
  item.command = "tabby.openSettings";
}

function toDisabled() {
  item.color = colorWarning;
  item.backgroundColor = backgroundColorWarning;
  item.text = `${iconDisabled} ${label}`;
  item.tooltip = "Tabby is disabled. Click to enable.";
  item.command = "tabby.toggleEnabled";
}
