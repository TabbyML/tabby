#!/bin/env node

import { TabbyAgent } from "./TabbyAgent";
import { StdIO } from "./StdIO";
import { LspServer } from "./LspServer";

const args = process.argv.slice(2);

let server;
if (args.indexOf("--lsp") >= 0) {
  server = new LspServer();
} else {
  server = new StdIO();
}
const agent = new TabbyAgent();
server.bind(agent);
server.listen();
