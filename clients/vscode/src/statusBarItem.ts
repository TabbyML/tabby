import { StatusBarAlignment, ThemeColor, window, workspace } from "vscode";
import { createMachine, interpret } from "@xstate/fsm";
import { agent } from "./agent";
import { notifications } from "./notifications";

const label = "Tabby";
const iconLoading = "$(loading~spin)";
const iconReady = "$(check)";
const iconDisconnected = "$(plug)";
const iconUnauthorized = "$(key)";
const iconDisabled = "$(x)";
const colorNormal = new ThemeColor("statusBar.foreground");
const colorWarning = new ThemeColor("statusBarItem.warningForeground");
const backgroundColorNormal = new ThemeColor("statusBar.background");
const backgroundColorWarning = new ThemeColor("statusBarItem.warningBackground");

const item = window.createStatusBarItem(StatusBarAlignment.Right);
const fsm = createMachine({
  id: "statusBarItem",
  initial: "loading",
  states: {
    loading: {
      on: { ready: "ready", disconnected: "disconnected", unauthorized: "unauthorized", disabled: "disabled" },
      entry: () => toLoading(),
    },
    ready: {
      on: { disconnected: "disconnected", unauthorized: "unauthorized", disabled: "disabled" },
      entry: () => toReady(),
    },
    disconnected: {
      on: { ready: "ready", unauthorized: "unauthorized", disabled: "disabled" },
      entry: () => toDisconnected(),
    },
    unauthorized: {
      on: {
        ready: "ready",
        disconnected: "disconnected",
        disabled: "disabled",
        openAuthPage: "unauthorizedAndAuthPageOpen",
      },
      entry: () => {
        toUnauthorized();
        notifications.showInformationStartAuth({
          onOpenAuthPage: () => {
            fsmService.send("openAuthPage");
          },
        });
      },
    },
    unauthorizedAndAuthPageOpen: {
      on: { ready: "ready", disconnected: "disconnected", disabled: "disabled" },
      exit: (_, event) => {
        if (event.type === "ready") {
          notifications.showInformationAuthSuccess();
        }
      },
    },
    disabled: {
      on: { loading: "loading", ready: "ready", disconnected: "disconnected", unauthorized: "unauthorized" },
      entry: () => toDisabled(),
    },
  },
});
const fsmService = interpret(fsm);

function toLoading() {
  item.color = colorNormal;
  item.backgroundColor = backgroundColorNormal;
  item.text = `${iconLoading} ${label}`;
  item.tooltip = "Tabby is initializing.";
  item.command = { title: "", command: "tabby.statusBarItemClicked", arguments: ["loading"] };
}

function toReady() {
  item.color = colorNormal;
  item.backgroundColor = backgroundColorNormal;
  item.text = `${iconReady} ${label}`;
  item.tooltip = "Tabby is providing code suggestions for you.";
  item.command = { title: "", command: "tabby.statusBarItemClicked", arguments: ["ready"] };
}

function toDisconnected() {
  item.color = colorWarning;
  item.backgroundColor = backgroundColorWarning;
  item.text = `${iconDisconnected} ${label}`;
  item.tooltip = "Cannot connect to Tabby Server. Click to open settings.";
  item.command = { title: "", command: "tabby.statusBarItemClicked", arguments: ["disconnected"] };
}

function toUnauthorized() {
  item.color = colorWarning;
  item.backgroundColor = backgroundColorWarning;
  item.text = `${iconUnauthorized} ${label}`;
  item.tooltip = "Tabby Server requires authorization. Click to continue.";
  item.command = { title: "", command: "tabby.statusBarItemClicked", arguments: ["unauthorized"] };
}

function toDisabled() {
  item.color = colorWarning;
  item.backgroundColor = backgroundColorWarning;
  item.text = `${iconDisabled} ${label}`;
  item.tooltip = "Tabby is disabled.";
  item.command = { title: "", command: "tabby.statusBarItemClicked", arguments: ["disabled"] };
}

function updateStatusBarItem() {
  const enabled = workspace.getConfiguration("tabby").get("codeCompletion", true);
  if (!enabled) {
    fsmService.send("disabled");
  } else {
    const status = agent().getStatus();
    switch (status) {
      case "notInitialized":
        fsmService.send("loading");
        break;
      case "ready":
      case "disconnected":
      case "unauthorized":
        fsmService.send(status);
        break;
    }
  }
}

export const tabbyStatusBarItem = () => {
  fsmService.start();
  updateStatusBarItem();

  workspace.onDidChangeConfiguration((event) => {
    if (event.affectsConfiguration("tabby")) {
      updateStatusBarItem();
    }
  });
  agent().on("statusChanged", updateStatusBarItem);

  item.show();
  return item;
};
