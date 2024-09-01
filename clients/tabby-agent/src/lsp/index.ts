#!/usr/bin/env node

import * as dns from "node:dns";
import { TabbyAgent } from "../TabbyAgent";
import { Server } from "./Server";
import { isBrowser } from "../env";

if (!isBrowser) {
  dns.setDefaultResultOrder("ipv4first");
}
const server = new Server(new TabbyAgent());
server.listen();
