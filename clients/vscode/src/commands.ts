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
import { agent } from "./agent";
import { notifications } from "./notifications";

const configTarget = ConfigurationTarget.Global;

type Command = {
  command: string;
  callback: (...args: any[]) => any;
  thisArg?: any;
};

const toggleInlineCompletionTriggerMode: Command = {
  command: "tabby.toggleInlineCompletionTriggerMode",
  callback: (value: "automatic" | "manual" | undefined) => {
    const configuration = workspace.getConfiguration("tabby");
    let target = value;
    if (!target) {
      const current = configuration.get("inlineCompletion.triggerMode", "automatic");
      if (current === "automatic") {
        target = "manual";
      } else {
        target = "automatic";
      }
    }
    configuration.update("inlineCompletion.triggerMode", target, configTarget, false);
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
    const agentUserConfig = Uri.joinPath(Uri.file(require("os").homedir()), ".tabby-client", "agent", "config.toml");
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
        const abortController = new AbortController();
        token.onCancellationRequested(() => {
          abortController.abort();
        });
        const signal = abortController.signal;
        try {
          callbacks?.onAuthStart?.();
          progress.report({ message: "Generating authorization url..." });
          let authUrl = await agent().requestAuthUrl({ signal });
          if (authUrl) {
            env.openExternal(Uri.parse(authUrl.authUrl));
            progress.report({ message: "Waiting for authorization from browser..." });
            await agent().waitForAuthToken(authUrl.code, { signal });
            assert(agent().getStatus() === "ready");
            notifications.showInformationAuthSuccess();
          } else if (agent().getStatus() === "ready") {
            notifications.showInformationWhenStartAuthButAlreadyAuthorized();
          } else {
            notifications.showInformationWhenAuthFailed();
          }
        } catch (error: any) {
          if (error.name === "AbortError") {
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
      case "initializing":
        notifications.showInformationWhenInitializing();
        break;
      case "automatic":
        notifications.showInformationWhenAutomaticTrigger();
        break;
      case "manual":
        notifications.showInformationWhenManualTrigger();
        break;
      case "loading":
        notifications.showInformationWhenManualTriggerLoading();
        break;
      case "disabled":
        notifications.showInformationWhenInlineSuggestDisabled();
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
            notifications.showInformationWhenSlowCompletionResponseTime();
            break;
          case "highCompletionTimeoutRate":
            notifications.showInformationWhenHighCompletionTimeoutRate();
            break;
        }
        break;
    }
  },
};

const triggerInlineCompletion: Command = {
  command: "tabby.inlineCompletion.trigger",
  callback: () => {
    commands.executeCommand("editor.action.inlineSuggest.trigger");
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
    toggleInlineCompletionTriggerMode,
    setApiEndpoint,
    openSettings,
    openTabbyAgentSettings,
    openKeybindings,
    gettingStarted,
    emitEvent,
    openAuthPage,
    statusBarItemClicked,
    triggerInlineCompletion,
    acceptInlineCompletion,
    acceptInlineCompletionNextWord,
    acceptInlineCompletionNextLine,
  ].map((command) => commands.registerCommand(command.command, command.callback, command.thisArg));
