import { Connection, ExecuteCommandParams } from "vscode-languageserver";
import {
  ServerCapabilities,
  StatusShowHelpMessageRequest,
  ChatEditResolveRequest,
  ChatEditResolveParams,
} from "./protocol";
import { ChatEditProvider } from "./ChatEditProvider";
import { StatusProvider } from "./StatusProvider";

export class CommandProvider {
  constructor(
    private readonly connection: Connection,
    private readonly chatEditProvider: ChatEditProvider,
    private readonly statusProvider: StatusProvider,
  ) {
    this.connection.onExecuteCommand(async (params) => {
      return this.executeCommand(params);
    });
  }

  fillServerCapabilities(capabilities: ServerCapabilities): void {
    capabilities.executeCommandProvider = {
      commands: [StatusShowHelpMessageRequest.method, ChatEditResolveRequest.method],
    };
  }

  async executeCommand(params: ExecuteCommandParams): Promise<void> {
    if (params.command === StatusShowHelpMessageRequest.method) {
      await this.statusProvider.showStatusHelpMessage(this.connection);
    } else if (params.command === ChatEditResolveRequest.method) {
      const commandParams = params.arguments?.[0] as ChatEditResolveParams;
      if (commandParams) {
        await this.chatEditProvider.resolveEdit(commandParams);
      }
    }
  }
}
