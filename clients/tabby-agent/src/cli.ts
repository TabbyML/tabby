#!/bin/env node

import { TabbyAgent } from "./TabbyAgent";
import { JsonLineServer } from "./JsonLineServer";
import { LspServer } from "./LspServer";

const args = process.argv.slice(2);

let server;
if (args.indexOf("--lsp") >= 0) {
  server = new LspServer();
} else {
  server = new JsonLineServer();
}
const agent = new TabbyAgent();
server.bind(agent);
server.listen();
