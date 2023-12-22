import {
  ConfigurationTarget,
  InputBoxValidationSeverity,
  ProgressLocation,
  Uri,
  ThemeIcon,
  ExtensionContext,
  workspace,
  window,
  env,
  commands,
} from "vscode";
import os from "os";
import { strict as assert } from "assert";
import { agent } from "./agent";
import { notifications } from "./notifications";
import { TabbyCompletionProvider } from "./TabbyCompletionProvider";
import { TabbyStatusBarItem } from "./TabbyStatusBarItem";

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
            const url = new URL(input);
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

const setApiToken = (context: ExtensionContext): Command => {
  return {
    command: "tabby.setApiToken",
    callback: () => {
      const currentToken = agent().getConfig()["server"]["token"].trim();
      window
        .showInputBox({
          prompt: "Enter your personal token",
          value: currentToken.length > 0 ? currentToken : undefined,
          password: true,
        })
        .then((token) => {
          if (token === undefined) {
            return; // User canceled
          }
          if (token.length > 0) {
            console.debug("Set token: ", token);
            context.globalState.update("server.token", token);
            agent().updateConfig("server.token", token);
          } else {
            console.debug("Clear token.");
            context.globalState.update("server.token", undefined);
            agent().clearConfig("server.token");
          }
        });
    },
  };
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
    const agentUserConfig = Uri.joinPath(Uri.file(os.homedir()), ".tabby-client", "agent", "config.toml");
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

/** @deprecated Tabby Cloud auth */
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
          const authUrl = await agent().requestAuthUrl({ signal });
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

const applyCallback: Command = {
  command: "tabby.applyCallback",
  callback: (callback) => {
    callback?.();
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

const acceptInlineCompletionNextWord = (completionProvider: TabbyCompletionProvider): Command => {
  return {
    command: "tabby.inlineCompletion.acceptNextWord",
    callback: () => {
      completionProvider.postEvent("accept_word");
      commands.executeCommand("editor.action.inlineSuggest.acceptNextWord");
    },
  };
};

const acceptInlineCompletionNextLine = (completionProvider: TabbyCompletionProvider): Command => {
  return {
    command: "tabby.inlineCompletion.acceptNextLine",
    callback: () => {
      completionProvider.postEvent("accept_line");
      // FIXME: this command move cursor to next line, but we want to move cursor to the end of current line
      commands.executeCommand("editor.action.inlineSuggest.acceptNextLine");
    },
  };
};

const openOnlineHelp: Command = {
  command: "tabby.openOnlineHelp",
  callback: () => {
    window
      .showQuickPick([
        {
          label: "Online Documentation",
          iconPath: new ThemeIcon("book"),
          alwaysShow: true,
        },
        {
          label: "Model Registry",
          description: "Explore more recommend models from Tabby's model registry",
          iconPath: new ThemeIcon("library"),
          alwaysShow: true,
        },
        {
          label: "Tabby Slack Community",
          description: "Join Tabby's Slack community to get help or feed back",
          iconPath: new ThemeIcon("comment-discussion"),
          alwaysShow: true,
        },
        {
          label: "Tabby GitHub Repository",
          description: "View the source code for Tabby, and open issues",
          iconPath: new ThemeIcon("github"),
          alwaysShow: true,
        },
      ])
      .then((selection) => {
        if (selection) {
          switch (selection.label) {
            case "Online Documentation":
              env.openExternal(Uri.parse("https://tabby.tabbyml.com/"));
              break;
            case "Model Registry":
              env.openExternal(Uri.parse("https://tabby.tabbyml.com/docs/models/"));
              break;
            case "Tabby Slack Community":
              env.openExternal(
                Uri.parse("https://join.slack.com/t/tabbycommunity/shared_invite/zt-1xeiddizp-bciR2RtFTaJ37RBxr8VxpA"),
              );
              break;
            case "Tabby GitHub Repository":
              env.openExternal(Uri.parse("https://github.com/tabbyml/tabby"));
              break;
          }
        }
      });
  },
};

const muteNotifications = (context: ExtensionContext, statusBarItem: TabbyStatusBarItem): Command => {
  return {
    command: "tabby.notifications.mute",
    callback: (type: string) => {
      const notifications = context.globalState.get<string[]>("notifications.muted", []);
      notifications.push(type);
      context.globalState.update("notifications.muted", notifications);
      statusBarItem.refresh();
    },
  };
};

const resetMutedNotifications = (context: ExtensionContext, statusBarItem: TabbyStatusBarItem): Command => {
  return {
    command: "tabby.notifications.resetMuted",
    callback: (type?: string) => {
      const notifications = context.globalState.get<string[]>("notifications.muted", []);
      if (type) {
        context.globalState.update(
          "notifications.muted",
          notifications.filter((t) => t !== type),
        );
      } else {
        context.globalState.update("notifications.muted", []);
      }
      statusBarItem.refresh();
    },
  };
};

export const tabbyCommands = (
  context: ExtensionContext,
  completionProvider: TabbyCompletionProvider,
  statusBarItem: TabbyStatusBarItem,
) =>
  [
    toggleInlineCompletionTriggerMode,
    setApiEndpoint,
    setApiToken(context),
    openSettings,
    openTabbyAgentSettings,
    openKeybindings,
    gettingStarted,
    openAuthPage,
    applyCallback,
    triggerInlineCompletion,
    acceptInlineCompletion,
    acceptInlineCompletionNextWord(completionProvider),
    acceptInlineCompletionNextLine(completionProvider),
    openOnlineHelp,
    muteNotifications(context, statusBarItem),
    resetMutedNotifications(context, statusBarItem),
  ].map((command) => commands.registerCommand(command.command, command.callback, command.thisArg));
