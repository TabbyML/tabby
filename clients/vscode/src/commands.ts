import {
  ConfigurationTarget,
  InputBoxValidationSeverity,
  QuickPickItem,
  QuickPickItemKind,
  Uri,
  workspace,
  window,
  env,
  commands,
} from "vscode";
import { strict as assert } from "assert";
import { Duration } from "@sapphire/duration";
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
    const enabled = configuration.get("enabled", true);
    console.debug(`Toggle Enabled: ${enabled} -> ${!enabled}.`);
    configuration.update("enabled", !enabled, configTarget, false);
  },
};

const setSuggestionDelay: Command = {
  command: "tabby.setSuggestionDelay",
  callback: () => {
    const configuration = workspace.getConfiguration("tabby");
    const current = configuration.get("suggestionDelay", 150);
    const items = {
      Immediately: 0, // ms
      Default: 150,
      Slowly: 1000,
    };
    const createQuickPickItem = (value: number): QuickPickItem => {
      const tags: string[] = [];
      if (value == current) {
        tags.push("Current");
      }
      Object.entries(items).forEach(([k, v]) => {
        if (v == value) {
          tags.push(k);
        }
      });
      return {
        label: value % 1000 == 0 ? `${value / 1000}s` : `${value}ms`,
        description: tags.join(" "),
        alwaysShow: true,
      };
    };
    const buildQuickPickList = (input: string = "") => {
      const list: QuickPickItem[] = [];
      const customized = new Duration(input).offset || Number.parseInt(input);
      if (customized >= 0) {
        list.push(createQuickPickItem(customized));
      }
      if (current != customized) {
        list.push(createQuickPickItem(current));
      }
      list.push({
        label: "",
        kind: QuickPickItemKind.Separator,
      });
      Object.values(items)
        .filter((item) => item != current && item != customized)
        .forEach((item) => list.push(createQuickPickItem(item)));
      return list;
    };
    const quickPick = window.createQuickPick();
    quickPick.placeholder = "Enter the delay after which the completion request is sent";
    quickPick.matchOnDescription = true;
    quickPick.items = buildQuickPickList();
    quickPick.onDidChangeValue((input: string) => {
      quickPick.items = buildQuickPickList(input);
    });
    quickPick.onDidAccept(() => {
      quickPick.hide();
      const delay = new Duration(quickPick.selectedItems[0].label).offset;
      console.debug("Set suggestion delay: ", delay);
      configuration.update("suggestionDelay", delay, configTarget, false);
    });
    quickPick.show();
  },
};

const setServerUrl: Command = {
  command: "tabby.setServerUrl",
  callback: () => {
    const configuration = workspace.getConfiguration("tabby");
    window
      .showInputBox({
        prompt: "Enter the URL of your Tabby Server",
        value: configuration.get("serverUrl", ""),
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
          configuration.update("serverUrl", url, configTarget, false);
        }
      });
  },
};

const openSettings: Command = {
  command: "tabby.openSettings",
  callback: () => {
    commands.executeCommand("workbench.action.openSettings", "tabby");
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
  callback: () => {
    agent()
      .startAuth()
      .then((authUrl) => {
        if (authUrl) {
          env.openExternal(Uri.parse(authUrl));
        }
      });
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
      case "disabled":
        notifications.showInformationWhenDisabled();
        break;
    }
  },
};

export const tabbyCommands = () =>
  [toggleEnabled, setServerUrl, setSuggestionDelay, openSettings, emitEvent, openAuthPage, statusBarItemClicked].map(
    (command) => commands.registerCommand(command.command, command.callback, command.thisArg)
  );
