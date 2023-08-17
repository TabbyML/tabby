import { commands, window, workspace, ConfigurationTarget } from "vscode";
import { agent } from "./agent";

function showInformationWhenLoading() {
  window.showInformationMessage("Tabby is initializing.", "Settings").then((selection) => {
    switch (selection) {
      case "Settings":
        commands.executeCommand("tabby.openSettings");
        break;
    }
  });
}

function showInformationWhenDisabled() {
  window
    .showInformationMessage("Tabby code completion is disabled. Enable it?", "Enable", "Settings")
    .then((selection) => {
      switch (selection) {
        case "Enable":
          commands.executeCommand("tabby.toggleEnabled");
          break;
        case "Settings":
          commands.executeCommand("tabby.openSettings");
          break;
      }
    });
}

function showInformationWhenReady() {
  window
    .showInformationMessage("Tabby is providing code suggestions for you. Disable it?", "Disable", "Settings")
    .then((selection) => {
      switch (selection) {
        case "Disable":
          commands.executeCommand("tabby.toggleEnabled");
          break;
        case "Settings":
          commands.executeCommand("tabby.openSettings");
          break;
      }
    });
}

function showInformationWhenDisconnected() {
  window
    .showInformationMessage("Cannot connect to Tabby Server. Please check settings.", "Settings")
    .then((selection) => {
      switch (selection) {
        case "Settings":
          commands.executeCommand("tabby.openSettings");
          break;
      }
    });
}

function showInformationStartAuth(callbacks?: { onAuthStart?: () => void; onAuthEnd?: () => void }) {
  window
    .showWarningMessage(
      "Tabby Server requires authorization. Continue to open authorization page in your browser.",
      "Continue",
      "Settings",
    )
    .then((selection) => {
      switch (selection) {
        case "Continue":
          commands.executeCommand("tabby.openAuthPage", callbacks);
          break;
        case "Settings":
          commands.executeCommand("tabby.openSettings");
      }
    });
}

function showInformationAuthSuccess() {
  window.showInformationMessage("Congrats, you're authorized, start to use Tabby now.");
}

function showInformationWhenStartAuthButAlreadyAuthorized() {
  window.showInformationMessage("You are already authorized now.");
}

function showInformationWhenAuthFailed() {
  window.showWarningMessage("Cannot connect to server. Please check settings.", "Settings").then((selection) => {
    switch (selection) {
      case "Settings":
        commands.executeCommand("tabby.openSettings");
        break;
    }
  });
}

function showInformationWhenInlineSuggestDisabled() {
  window
    .showWarningMessage(
      "Tabby's suggestion is not showing because inline suggestion is disabled. Please enable it first.",
      "Enable",
      "Settings",
    )
    .then((selection) => {
      switch (selection) {
        case "Enable":
          const configuration = workspace.getConfiguration("editor");
          console.debug(`Set editor.inlineSuggest.enabled: true.`);
          configuration.update("inlineSuggest.enabled", true, ConfigurationTarget.Global, false);
          break;
        case "Settings":
          commands.executeCommand("workbench.action.openSettings", "@id:editor.inlineSuggest.enabled");
          break;
      }
    });
}

const helpMessageForCompletionResponseTimeIssue = `Possible causes of this issue are:
 - Server overload or running a large model on CPU. Please contact your Tabby server administrator for assistance.
 - A poor network connection. Please check your network and proxy settings.`;

function showInformationWhenSlowCompletionResponseTime(modal: boolean = false) {
  if (modal) {
    const stats = agent()
      .getIssues()
      .find((issue) => issue.name === "slowCompletionResponseTime")?.completionResponseStats;
    let statsMessage = "";
    if (stats && stats["responses"] && stats["averageResponseTime"]) {
      statsMessage = `The average response time of recent ${stats["responses"]} completion requests is ${Number(
        stats["averageResponseTime"],
      ).toFixed(0)}ms.\n`;
    }
    window.showWarningMessage("Completion requests appear to take too much time.", {
      modal: true,
      detail: statsMessage + helpMessageForCompletionResponseTimeIssue,
    });
  } else {
    window
      .showWarningMessage("Completion requests appear to take too much time.", "Detail")
      .then((selection) => {
        switch (selection) {
          case "Detail":
            showInformationWhenSlowCompletionResponseTime(true);
            break;
        }
      });
  }
}

function showInformationWhenHighCompletionTimeoutRate(modal: boolean = false) {
  if (modal) {
    const stats = agent()
      .getIssues()
      .find((issue) => issue.name === "highCompletionTimeoutRate")?.completionResponseStats;
    let statsMessage = "";
    if (stats && stats["total"] && stats["timeouts"]) {
      statsMessage = `${stats["timeouts"]} of ${stats["total"]} completion requests timed out.\n`;
    }
    window.showWarningMessage("Most completion requests timed out.", {
      modal: true,
      detail: statsMessage + helpMessageForCompletionResponseTimeIssue,
    });
  } else {
    window.showWarningMessage("Most completion requests timed out.", "Detail").then((selection) => {
      switch (selection) {
        case "Detail":
          showInformationWhenHighCompletionTimeoutRate(true);
          break;
      }
    });
  }
}

export const notifications = {
  showInformationWhenLoading,
  showInformationWhenDisabled,
  showInformationWhenReady,
  showInformationWhenDisconnected,
  showInformationStartAuth,
  showInformationAuthSuccess,
  showInformationWhenStartAuthButAlreadyAuthorized,
  showInformationWhenAuthFailed,
  showInformationWhenInlineSuggestDisabled,
  showInformationWhenSlowCompletionResponseTime,
  showInformationWhenHighCompletionTimeoutRate,
};
