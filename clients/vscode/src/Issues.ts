import { EventEmitter } from "events";
import { commands, window } from "vscode";
import { IssueName, IssueDetailResult as IssueDetail } from "tabby-agent";
import { Client } from "./lsp/Client";
import { Config } from "./Config";

export class Issues extends EventEmitter {
  private issues: IssueName[] = [];

  constructor(
    private readonly client: Client,
    private readonly config: Config,
  ) {
    super();
    // schedule initial fetch
    this.client.agent.fetchIssues().then((params) => {
      this.issues = params.issues;
      this.emit("updated");
    });
    this.client.agent.on("didUpdateIssues", (params: IssueName[]) => {
      this.issues = params;
      this.emit("updated");
    });
  }

  get filteredIssues(): IssueName[] {
    return this.issues.filter((item) => !this.config.mutedNotifications.includes(item));
  }

  get first(): IssueName | undefined {
    return this.filteredIssues[0];
  }

  get current(): IssueName[] {
    return this.filteredIssues;
  }

  get length(): number {
    return this.filteredIssues.length;
  }

  async fetchDetail(issue: IssueName): Promise<IssueDetail> {
    return await this.client.agent.fetchIssueDetail({ name: issue, helpMessageFormat: "markdown" });
  }

  async showHelpMessage(issue?: IssueName | undefined, modal = false) {
    const name = issue ?? this.first;
    if (!name) {
      return;
    }
    if (name === "connectionFailed") {
      if (modal) {
        const detail = await this.client.agent.fetchIssueDetail({
          name: "connectionFailed",
          helpMessageFormat: "markdown",
        });
        window
          .showWarningMessage(
            `Cannot connect to Tabby Server.`,
            {
              modal: true,
              detail: detail.helpMessage,
            },
            "Settings",
            "Online Help...",
          )
          .then((selection) => {
            switch (selection) {
              case "Online Help...":
                commands.executeCommand("tabby.openOnlineHelp");
                break;
              case "Settings":
                commands.executeCommand("tabby.openSettings");
                break;
            }
          });
      } else {
        window.showWarningMessage(`Cannot connect to Tabby Server.`, "Detail...", "Settings").then((selection) => {
          switch (selection) {
            case "Detail...":
              this.showHelpMessage(name, true);
              break;
            case "Settings":
              commands.executeCommand("tabby.openSettings");
              break;
          }
        });
      }
    }
    if (name === "highCompletionTimeoutRate") {
      if (modal) {
        const detail = await this.client.agent.fetchIssueDetail({
          name: "connectionFailed",
          helpMessageFormat: "markdown",
        });
        window
          .showWarningMessage(
            "Most completion requests timed out.",
            {
              modal: true,
              detail: detail.helpMessage,
            },
            "Online Help...",
            "Don't Show Again",
          )
          .then((selection) => {
            switch (selection) {
              case "Online Help...":
                commands.executeCommand("tabby.openOnlineHelp");
                break;
              case "Don't Show Again":
                commands.executeCommand("tabby.notifications.mute", "completionResponseTimeIssues");
                break;
            }
          });
      } else {
        window
          .showWarningMessage("Most completion requests timed out.", "Detail...", "Settings", "Don't Show Again")
          .then((selection) => {
            switch (selection) {
              case "Detail...":
                this.showHelpMessage(name, true);
                break;
              case "Settings":
                commands.executeCommand("tabby.openSettings");
                break;
              case "Don't Show Again":
                commands.executeCommand("tabby.notifications.mute", "completionResponseTimeIssues");
                break;
            }
          });
      }
    }
    if (name === "slowCompletionResponseTime") {
      if (modal) {
        const detail = await this.client.agent.fetchIssueDetail({
          name: "slowCompletionResponseTime",
          helpMessageFormat: "markdown",
        });
        window
          .showWarningMessage(
            "Completion requests appear to take too much time.",
            {
              modal: true,
              detail: detail.helpMessage,
            },
            "Online Help...",
            "Don't Show Again",
          )
          .then((selection) => {
            switch (selection) {
              case "Online Help...":
                commands.executeCommand("tabby.openOnlineHelp");
                break;
              case "Don't Show Again":
                commands.executeCommand("tabby.notifications.mute", "completionResponseTimeIssues");
                break;
            }
          });
      } else {
        window
          .showWarningMessage(
            "Completion requests appear to take too much time.",
            "Detail",
            "Settings",
            "Don't Show Again",
          )
          .then((selection) => {
            switch (selection) {
              case "Detail":
                this.showHelpMessage(name, true);
                break;
              case "Settings":
                commands.executeCommand("tabby.openSettings");
                break;
              case "Don't Show Again":
                commands.executeCommand("tabby.notifications.mute", "completionResponseTimeIssues");
                break;
            }
          });
      }
    }
  }
}
