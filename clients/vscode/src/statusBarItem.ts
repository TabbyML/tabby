import { StatusBarAlignment, ThemeColor, window, workspace } from "vscode";
import { createMachine, interpret } from "@xstate/fsm";
import { agent } from "./agent";
import { notifications } from "./notifications";

const label = "Tabby";
const iconLoading = "$(loading~spin)";
const iconReady = "$(check)";
const iconDisconnected = "$(plug)";
const iconUnauthorized = "$(key)";
const iconIssueExist = "$(warning)";
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
      on: {
        ready: "ready",
        disconnected: "disconnected",
        unauthorized: "unauthorized",
        issuesExist: "issuesExist",
        disabled: "disabled",
      },
      entry: () => toLoading(),
    },
    ready: {
      on: {
        loading: "loading",
        disconnected: "disconnected",
        unauthorized: "unauthorized",
        issuesExist: "issuesExist",
        disabled: "disabled",
      },
      entry: () => toReady(),
    },
    disconnected: {
      on: {
        loading: "loading",
        ready: "ready",
        unauthorized: "unauthorized",
        issuesExist: "issuesExist",
        disabled: "disabled",
      },
      entry: () => toDisconnected(),
    },
    unauthorized: {
      on: {
        ready: "ready",
        disconnected: "disconnected",
        disabled: "disabled",
        issuesExist: "issuesExist",
        authStart: "unauthorizedAndAuthInProgress",
      },
      entry: () => toUnauthorized(),
    },
    unauthorizedAndAuthInProgress: {
      on: {
        ready: "ready",
        disconnected: "disconnected",
        issuesExist: "issuesExist",
        disabled: "disabled",
        authEnd: "unauthorized", // if auth succeeds, we will get `ready` before `authEnd` event
      },
      entry: () => toUnauthorizedAndAuthInProgress(),
    },
    issuesExist: {
      on: {
        loading: "loading",
        ready: "ready",
        disconnected: "disconnected",
        unauthorized: "unauthorized",
        disabled: "disabled",
      },
      entry: () => toIssuesExist(),
    },
    disabled: {
      on: {
        loading: "loading",
        ready: "ready",
        disconnected: "disconnected",
        unauthorized: "unauthorized",
        issuesExist: "issuesExist",
      },
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

function toUnauthorizedAndAuthInProgress() {
  item.color = colorWarning;
  item.backgroundColor = backgroundColorWarning;
  item.text = `${iconUnauthorized} ${label}`;
  item.tooltip = "Waiting for authorization.";
  item.command = undefined;
}

function toIssuesExist() {
  item.color = colorWarning;
  item.backgroundColor = backgroundColorWarning;
  item.text = `${iconIssueExist} ${label}`;
  switch (agent().getIssues()[0]?.name) {
    case "slowCompletionResponseTime":
      item.tooltip = "Completion requests appear to take too much time.";
      break;
    case "highCompletionTimeoutRate":
      item.tooltip = "Most completion requests timed out.";
      break;
    default:
      item.tooltip = "";
      break;
  }
  item.command = { title: "", command: "tabby.statusBarItemClicked", arguments: ["issuesExist"] };
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
  const inlineSuggestEnabled = workspace.getConfiguration("editor").get("inlineSuggest.enabled", true);
  if (!enabled || !inlineSuggestEnabled) {
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
      case "issuesExist":
        fsmService.send(status);
        break;
    }
  }
}

export const tabbyStatusBarItem = () => {
  fsmService.start();
  updateStatusBarItem();

  workspace.onDidChangeConfiguration((event) => {
    if (event.affectsConfiguration("tabby") || event.affectsConfiguration("editor.inlineSuggest")) {
      updateStatusBarItem();
    }
  });
  agent().on("statusChanged", updateStatusBarItem);

  agent().on("authRequired", () => {
    notifications.showInformationStartAuth({
      onAuthStart: () => {
        fsmService.send("authStart");
      },
      onAuthEnd: () => {
        fsmService.send("authEnd");
      },
    });
  });

  agent().on("newIssue", (event) => {
    if (event.issue.name === "slowCompletionResponseTime") {
      notifications.showInformationWhenSlowCompletionResponseTime();
    } else if (event.issue.name === "highCompletionTimeoutRate") {
      notifications.showInformationWhenHighCompletionTimeoutRate();
    }
  });

  item.show();
  return item;
};
