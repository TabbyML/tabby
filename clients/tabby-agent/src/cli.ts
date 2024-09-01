#!/usr/bin/env node

import { TabbyAgent } from "./TabbyAgent";
import { JsonLineServer } from "./JsonLineServer";
import { Server as LspServer } from "./lsp/Server";

const args = process.argv.slice(2);

const agent = new TabbyAgent();
let server;
if (args.indexOf("--lsp") >= 0) {
  server = new LspServer(agent);
} else {
  server = new JsonLineServer();
  server.bind(agent);
}
server.listen();
