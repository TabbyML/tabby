import { ConfigurationMiddleware as VscodeLspConfigurationMiddleware } from "vscode-languageclient";
import { ClientProvidedConfig } from "tabby-agent";
import { Config } from "../Config";

export class ConfigurationMiddleware implements VscodeLspConfigurationMiddleware {
  constructor(private readonly config: Config) {}

  async configuration(): Promise<ClientProvidedConfig[]> {
    return [this.config.buildClientProvidedConfig()];
  }
}
