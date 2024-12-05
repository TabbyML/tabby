#!/usr/bin/env node

import * as dns from "node:dns";
import { isBrowser } from "./env";
import { Server } from "./server";

if (!isBrowser) {
  dns.setDefaultResultOrder("ipv4first");
}

const server = new Server();
server.listen();
