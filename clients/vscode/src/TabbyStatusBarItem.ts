import { StatusBarAlignment, ThemeColor, ExtensionContext, window } from "vscode";
import { createMachine, interpret } from "@xstate/fsm";
import type { StatusChangedEvent, AuthRequiredEvent, IssuesUpdatedEvent } from "tabby-agent";
import { agent } from "./agent";
import { notifications } from "./notifications";
import { TabbyCompletionProvider } from "./TabbyCompletionProvider";

const label = "Tabby";
const iconLoading = "$(loading~spin)";
const iconAutomatic = "$(check)";
const iconManual = "$(chevron-right)";
const iconDisabled = "$(x)";
const iconDisconnected = "$(plug)";
const iconUnauthorized = "$(key)";
const iconIssueExist = "$(warning)";
const colorNormal = new ThemeColor("statusBar.foreground");
const colorWarning = new ThemeColor("statusBarItem.warningForeground");
const backgroundColorNormal = new ThemeColor("statusBar.background");
const backgroundColorWarning = new ThemeColor("statusBarItem.warningBackground");

export class TabbyStatusBarItem {
  private item = window.createStatusBarItem(StatusBarAlignment.Right);
  private extensionContext: ExtensionContext;
  private completionProvider: TabbyCompletionProvider;
  private completionResponseWarningShown = false;

  private subStatusForReady = [
    {
      target: "issuesExist",
      cond: () => {
        let issues = agent().getIssues();
        if (
          this.extensionContext.globalState
            .get<string[]>("notifications.muted", [])
            .includes("completionResponseTimeIssues")
        ) {
          issues = issues.filter(
            (issue) => issue !== "highCompletionTimeoutRate" && issue !== "slowCompletionResponseTime",
          );
        }
        return issues.length > 0;
      },
    },
    {
      target: "automatic",
      cond: () => this.completionProvider.getTriggerMode() === "automatic" && !this.completionProvider.isLoading(),
    },
    {
      target: "manual",
      cond: () => this.completionProvider.getTriggerMode() === "manual" && !this.completionProvider.isLoading(),
    },
    {
      target: "loading",
      cond: () => this.completionProvider.isLoading(),
    },
    {
      target: "disabled",
      cond: () => this.completionProvider.getTriggerMode() === "disabled",
    },
  ];

  private fsm = createMachine({
    id: "statusBarItem",
    initial: "initializing",
    states: {
      initializing: {
        on: {
          ready: this.subStatusForReady,
          disconnected: "disconnected",
          unauthorized: "unauthorized",
        },
        entry: () => this.toInitializing(),
      },
      automatic: {
        on: {
          ready: this.subStatusForReady,
          disconnected: "disconnected",
          unauthorized: "unauthorized",
        },
        entry: () => this.toAutomatic(),
      },
      manual: {
        on: {
          ready: this.subStatusForReady,
          disconnected: "disconnected",
          unauthorized: "unauthorized",
        },
        entry: () => this.toManual(),
      },
      loading: {
        on: {
          ready: this.subStatusForReady,
          disconnected: "disconnected",
          unauthorized: "unauthorized",
        },
        entry: () => this.toLoading(),
      },
      disabled: {
        on: {
          ready: this.subStatusForReady,
          disconnected: "disconnected",
          unauthorized: "unauthorized",
        },
        entry: () => this.toDisabled(),
      },
      disconnected: {
        on: {
          ready: this.subStatusForReady,
          unauthorized: "unauthorized",
        },
        entry: () => this.toDisconnected(),
      },
      unauthorized: {
        on: {
          ready: this.subStatusForReady,
          disconnected: "disconnected",
        },
        entry: () => this.toUnauthorized(),
      },
      issuesExist: {
        on: {
          ready: this.subStatusForReady,
          disconnected: "disconnected",
          unauthorized: "unauthorized",
        },
        entry: () => this.toIssuesExist(),
      },
    },
  });

  private fsmService = interpret(this.fsm);

  constructor(context: ExtensionContext, completionProvider: TabbyCompletionProvider) {
    this.extensionContext = context;
    this.completionProvider = completionProvider;
    this.fsmService.start();
    this.fsmService.send(agent().getStatus());
    this.item.show();

    this.completionProvider.on("triggerModeUpdated", () => {
      this.refresh();
    });
    this.completionProvider.on("loadingStatusUpdated", () => {
      this.refresh();
    });

    agent().on("statusChanged", (event: StatusChangedEvent) => {
      console.debug("Tabby agent statusChanged", { event });
      this.fsmService.send(event.status);
    });

    agent().on("authRequired", (event: AuthRequiredEvent) => {
      console.debug("Tabby agent authRequired", { event });
      notifications.showInformationWhenUnauthorized();
    });

    agent().on("issuesUpdated", (event: IssuesUpdatedEvent) => {
      console.debug("Tabby agent issuesUpdated", { event });
      const status = agent().getStatus();
      this.fsmService.send(status);
      const showCompletionResponseWarnings =
        !this.completionResponseWarningShown &&
        !this.extensionContext.globalState
          .get<string[]>("notifications.muted", [])
          .includes("completionResponseTimeIssues");
      if (event.issues.includes("connectionFailed")) {
        // Only show this notification when user modifies the settings, do not show it when initializing
        // FIXME: refactor this use a flag marks the event is trigger by modifying settings or initializing
        if (status !== "notInitialized") {
          notifications.showInformationWhenDisconnected();
        }
      } else if (showCompletionResponseWarnings && event.issues.includes("highCompletionTimeoutRate")) {
        this.completionResponseWarningShown = true;
        notifications.showInformationWhenHighCompletionTimeoutRate();
      } else if (showCompletionResponseWarnings && event.issues.includes("slowCompletionResponseTime")) {
        this.completionResponseWarningShown = true;
        notifications.showInformationWhenSlowCompletionResponseTime();
      }
    });
  }

  public register() {
    return this.item;
  }

  public refresh() {
    this.fsmService.send(agent().getStatus());
  }

  private toInitializing() {
    this.item.color = colorNormal;
    this.item.backgroundColor = backgroundColorNormal;
    this.item.text = `${iconLoading} ${label}`;
    this.item.tooltip = "Tabby is initializing.";
    this.item.command = {
      title: "",
      command: "tabby.applyCallback",
      arguments: [() => notifications.showInformationWhenInitializing()],
    };
  }

  private toAutomatic() {
    this.item.color = colorNormal;
    this.item.backgroundColor = backgroundColorNormal;
    this.item.text = `${iconAutomatic} ${label}`;
    this.item.tooltip = "Tabby automatic code completion is enabled.";
    this.item.command = {
      title: "",
      command: "tabby.applyCallback",
      arguments: [() => notifications.showInformationWhenAutomaticTrigger()],
    };
  }

  private toManual() {
    this.item.color = colorNormal;
    this.item.backgroundColor = backgroundColorNormal;
    this.item.text = `${iconManual} ${label}`;
    this.item.tooltip = "Tabby is standing by, click or press `Alt + \\` to trigger code completion.";
    this.item.command = {
      title: "",
      command: "tabby.applyCallback",
      arguments: [() => notifications.showInformationWhenManualTrigger()],
    };
  }

  private toLoading() {
    this.item.color = colorNormal;
    this.item.backgroundColor = backgroundColorNormal;
    this.item.text = `${iconLoading} ${label}`;
    this.item.tooltip = "Tabby is generating code completions.";
    this.item.command = {
      title: "",
      command: "tabby.applyCallback",
      arguments: [() => notifications.showInformationWhenManualTriggerLoading()],
    };
  }

  private toDisabled() {
    this.item.color = colorWarning;
    this.item.backgroundColor = backgroundColorWarning;
    this.item.text = `${iconDisabled} ${label}`;
    this.item.tooltip = "Tabby is disabled. Click to check settings.";
    this.item.command = {
      title: "",
      command: "tabby.applyCallback",
      arguments: [() => notifications.showInformationWhenInlineSuggestDisabled()],
    };

    console.debug("Tabby code completion is enabled but inline suggest is disabled.");
    notifications.showInformationWhenInlineSuggestDisabled();
  }

  private toDisconnected() {
    this.item.color = colorWarning;
    this.item.backgroundColor = backgroundColorWarning;
    this.item.text = `${iconDisconnected} ${label}`;
    this.item.tooltip = "Cannot connect to Tabby Server. Click to open settings.";
    this.item.command = {
      title: "",
      command: "tabby.applyCallback",
      arguments: [() => notifications.showInformationWhenDisconnected()],
    };
  }

  private toUnauthorized() {
    this.item.color = colorWarning;
    this.item.backgroundColor = backgroundColorWarning;
    this.item.text = `${iconUnauthorized} ${label}`;
    this.item.tooltip = "Tabby Server requires authorization. Please set your personal token.";
    this.item.command = {
      title: "",
      command: "tabby.applyCallback",
      arguments: [() => notifications.showInformationWhenUnauthorized()],
    };
  }

  private toIssuesExist() {
    this.item.color = colorWarning;
    this.item.backgroundColor = backgroundColorWarning;
    this.item.text = `${iconIssueExist} ${label}`;
    const issue =
      agent().getIssueDetail({ name: "highCompletionTimeoutRate" }) ??
      agent().getIssueDetail({ name: "slowCompletionResponseTime" });
    switch (issue?.name) {
      case "highCompletionTimeoutRate":
        this.item.tooltip = "Most completion requests timed out.";
        break;
      case "slowCompletionResponseTime":
        this.item.tooltip = "Completion requests appear to take too much time.";
        break;
      default:
        this.item.tooltip = "";
        break;
    }
    this.item.command = {
      title: "",
      command: "tabby.applyCallback",
      arguments: [
        () => {
          switch (issue?.name) {
            case "highCompletionTimeoutRate":
              notifications.showInformationWhenHighCompletionTimeoutRate();
              break;
            case "slowCompletionResponseTime":
              notifications.showInformationWhenSlowCompletionResponseTime();
              break;
          }
        },
      ],
    };
  }
}
