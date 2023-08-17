import {
  ConfigurationTarget,
  InputBoxValidationSeverity,
  ProgressLocation,
  Uri,
  workspace,
  window,
  env,
  commands,
} from "vscode";
import { strict as assert } from "assert";
import { CancelablePromise } from "tabby-agent";
import { agent } from "./agent";
import { notifications } from "./notifications";

const configTarget = ConfigurationTarget.Global;

type Command = {
  command: string;
  callback: (...args: any[]) => any;
  thisArg?: any;
};

const toggleEnabled: Command = {
  command: "tabby.toggleEnabled",
  callback: () => {
    const configuration = workspace.getConfiguration("tabby");
    const enabled = configuration.get("codeCompletion", true);
    console.debug(`Toggle Enabled: ${enabled} -> ${!enabled}.`);
    configuration.update("codeCompletion", !enabled, configTarget, false);
  },
};

const setApiEndpoint: Command = {
  command: "tabby.setApiEndpoint",
  callback: () => {
    const configuration = workspace.getConfiguration("tabby");
    window
      .showInputBox({
        prompt: "Enter the URL of your Tabby Server",
        value: configuration.get("api.endpoint", ""),
        validateInput: (input: string) => {
          try {
            let url = new URL(input);
            assert(url.protocol == "http:" || url.protocol == "https:");
          } catch (_) {
            return {
              message: "Please enter a validate http or https URL.",
              severity: InputBoxValidationSeverity.Error,
            };
          }
          return null;
        },
      })
      .then((url) => {
        if (url) {
          console.debug("Set Tabby Server URL: ", url);
          configuration.update("api.endpoint", url, configTarget, false);
        }
      });
  },
};

const openSettings: Command = {
  command: "tabby.openSettings",
  callback: () => {
    commands.executeCommand("workbench.action.openSettings", "@ext:TabbyML.vscode-tabby");
  },
};

const openTabbyAgentSettings: Command = {
  command: "tabby.openTabbyAgentSettings",
  callback: () => {
    if (env.appHost !== "desktop") {
      window.showWarningMessage("Tabby Agent config file is not supported on web.", { modal: true });
      return;
    }
    const agentUserConfig = Uri.joinPath(Uri.file(require("os").homedir()), ".tabby", "agent", "config.toml");
    workspace.fs.stat(agentUserConfig).then(
      () => {
        workspace.openTextDocument(agentUserConfig).then((document) => {
          window.showTextDocument(document);
        });
      },
      () => {
        window.showWarningMessage("Tabby Agent config file not found.", { modal: true });
      },
    );
  },
};

const openKeybindings: Command = {
  command: "tabby.openKeybindings",
  callback: () => {
    commands.executeCommand("workbench.action.openGlobalKeybindings", "tabby.inlineCompletion");
  },
};

const gettingStarted: Command = {
  command: "tabby.gettingStarted",
  callback: () => {
    commands.executeCommand("workbench.action.openWalkthrough", "TabbyML.vscode-tabby#gettingStarted");
  },
};

const emitEvent: Command = {
  command: "tabby.emitEvent",
  callback: (event) => {
    console.debug("Emit Event: ", event);
    agent().postEvent(event);
  },
};

const openAuthPage: Command = {
  command: "tabby.openAuthPage",
  callback: (callbacks?: { onAuthStart?: () => void; onAuthEnd?: () => void }) => {
    window.withProgress(
      {
        location: ProgressLocation.Notification,
        title: "Tabby Server Authorization",
        cancellable: true,
      },
      async (progress, token) => {
        let requestAuthUrl: CancelablePromise<{ authUrl: string; code: string } | null>;
        let waitForAuthToken: CancelablePromise<any>;
        token.onCancellationRequested(() => {
          requestAuthUrl?.cancel();
          waitForAuthToken?.cancel();
        });
        try {
          callbacks?.onAuthStart?.();
          progress.report({ message: "Generating authorization url..." });
          requestAuthUrl = agent().requestAuthUrl();
          let authUrl = await requestAuthUrl;
          if (authUrl) {
            env.openExternal(Uri.parse(authUrl.authUrl));
            progress.report({ message: "Waiting for authorization from browser..." });
            waitForAuthToken = agent().waitForAuthToken(authUrl.code);
            await waitForAuthToken;
            assert(agent().getStatus() === "ready");
            notifications.showInformationAuthSuccess();
          } else if (agent().getStatus() === "ready") {
            notifications.showInformationWhenStartAuthButAlreadyAuthorized();
          } else {
            notifications.showInformationWhenAuthFailed();
          }
        } catch (error: any) {
          if (error.isCancelled) {
            return;
          }
          console.debug("Error auth", { error });
          notifications.showInformationWhenAuthFailed();
        } finally {
          callbacks?.onAuthEnd?.();
        }
      },
    );
  },
};

const statusBarItemClicked: Command = {
  command: "tabby.statusBarItemClicked",
  callback: (status) => {
    switch (status) {
      case "loading":
        notifications.showInformationWhenLoading();
        break;
      case "ready":
        notifications.showInformationWhenReady();
        break;
      case "disconnected":
        notifications.showInformationWhenDisconnected();
        break;
      case "unauthorized":
        notifications.showInformationStartAuth();
        break;
      case "issuesExist":
        switch (agent().getIssues()[0]?.name) {
          case "slowCompletionResponseTime":
            notifications.showInformationWhenSlowCompletionResponseTime(true);
            break;
          case "highCompletionTimeoutRate":
            notifications.showInformationWhenHighCompletionTimeoutRate(true);
            break;
        }
        break;
      case "disabled":
        const enabled = workspace.getConfiguration("tabby").get("codeCompletion", true);
        const inlineSuggestEnabled = workspace.getConfiguration("editor").get("inlineSuggest.enabled", true);
        if (enabled && !inlineSuggestEnabled) {
          notifications.showInformationWhenInlineSuggestDisabled();
        } else {
          notifications.showInformationWhenDisabled();
        }
        break;
    }
  },
};

const acceptInlineCompletion: Command = {
  command: "tabby.inlineCompletion.accept",
  callback: () => {
    commands.executeCommand("editor.action.inlineSuggest.commit");
  },
};

const acceptInlineCompletionNextWord: Command = {
  command: "tabby.inlineCompletion.acceptNextWord",
  callback: () => {
    // FIXME: sent event when partially accept?
    commands.executeCommand("editor.action.inlineSuggest.acceptNextWord");
  },
};

const acceptInlineCompletionNextLine: Command = {
  command: "tabby.inlineCompletion.acceptNextLine",
  callback: () => {
    // FIXME: sent event when partially accept?
    commands.executeCommand("editor.action.inlineSuggest.acceptNextLine");
  },
};

export const tabbyCommands = () =>
  [
    toggleEnabled,
    setApiEndpoint,
    openSettings,
    openTabbyAgentSettings,
    openKeybindings,
    gettingStarted,
    emitEvent,
    openAuthPage,
    statusBarItemClicked,
    acceptInlineCompletion,
    acceptInlineCompletionNextWord,
    acceptInlineCompletionNextLine,
  ].map((command) => commands.registerCommand(command.command, command.callback, command.thisArg));
