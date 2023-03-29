import { ConfigurationTarget, workspace, window, commands } from "vscode";
import { TabbyClient } from "./TabbyClient";

const target = ConfigurationTarget.Global;

type Command = {
  command: string;
  callback: (...args: any[]) => any;
  thisArg?: any;
};

const toogleEnabled: Command = {
  command: "tabby.toggleEnabled",
  callback: () => {
    const configuration = workspace.getConfiguration("tabby");
    const enabled = configuration.get("enabled", true);
    console.debug(`Toggle Enabled: ${enabled} -> ${!enabled}.`);
    configuration.update("enabled", !enabled, target, false);
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
      })
      .then((url) => {
        if (url) {
          console.debug("Set Tabby Server URL: ", url);
          configuration.update("serverUrl", url, target, false);
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

const tabbyClient = TabbyClient.getInstance();
const emitEvent: Command = {
  command: "tabby.emitEvent",
  callback: (event) => {
    console.debug("Emit Event: ", event);
    tabbyClient.postEvent(event);
  },
};

export const tabbyCommands = [toogleEnabled, setServerUrl, openSettings, emitEvent].map((command) =>
  commands.registerCommand(command.command, command.callback, command.thisArg)
);
