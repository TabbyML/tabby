import { Connection, ExecuteCommandParams } from "vscode-languageserver";
import { ServerCapabilities, ChatEditResolveParams } from "./protocol";
import { ChatEditProvider } from "./ChatEditProvider";

export class CommandProvider {
  constructor(
    private readonly connection: Connection,
    private readonly chatEditProvider: ChatEditProvider,
  ) {
    this.connection.onExecuteCommand(async (params) => {
      return this.executeCommand(params);
    });
  }

  fillServerCapabilities(capabilities: ServerCapabilities): void {
    capabilities.executeCommandProvider = {
      commands: ["tabby/chat/edit/resolve"],
    };
  }

  async executeCommand(params: ExecuteCommandParams): Promise<void> {
    if (params.command === "tabby/chat/edit/resolve") {
      const resolveParams = params.arguments?.[0] as ChatEditResolveParams;
      if (resolveParams) {
        await this.chatEditProvider.resolveEdit(resolveParams);
      }
    }
  }
}
