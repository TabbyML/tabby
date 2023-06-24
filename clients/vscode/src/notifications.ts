import { commands, window, workspace, ConfigurationTarget } from "vscode";

const configTarget = ConfigurationTarget.Global;

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
  window.showInformationMessage("Tabby is disabled. Enable it?", "Enable", "Settings").then((selection) => {
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
      "Settings"
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

export const notifications = {
  showInformationWhenLoading,
  showInformationWhenDisabled,
  showInformationWhenReady,
  showInformationWhenDisconnected,
  showInformationStartAuth,
  showInformationAuthSuccess,
  showInformationWhenStartAuthButAlreadyAuthorized,
  showInformationWhenAuthFailed,
};
