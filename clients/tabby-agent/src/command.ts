import { Feature } from "./feature";
import { Connection, ExecuteCommandParams } from "vscode-languageserver";
import {
  ServerCapabilities,
  StatusShowHelpMessageRequest,
  ChatEditResolveRequest,
  ChatEditResolveParams,
} from "./protocol";
import { ChatEditProvider } from "./chat/inlineEdit";
import { StatusProvider } from "./status";

export class CommandProvider implements Feature {
  constructor(
    private readonly chatEditProvider: ChatEditProvider,
    private readonly statusProvider: StatusProvider,
  ) {}

  initialize(connection: Connection): ServerCapabilities {
    connection.onExecuteCommand(async (params) => {
      return this.executeCommand(params);
    });
    return {
      executeCommandProvider: {
        commands: [StatusShowHelpMessageRequest.method, ChatEditResolveRequest.method],
      },
    };
  }

  async executeCommand(params: ExecuteCommandParams): Promise<void> {
    if (params.command === StatusShowHelpMessageRequest.method) {
      await this.statusProvider.showStatusHelpMessage();
    } else if (params.command === ChatEditResolveRequest.method) {
      const commandParams = params.arguments?.[0] as ChatEditResolveParams;
      if (commandParams) {
        await this.chatEditProvider.resolveEdit(commandParams);
      }
    }
  }
}
