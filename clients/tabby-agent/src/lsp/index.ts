#!/usr/bin/env node

import { TabbyAgent } from "../TabbyAgent";
import { Server } from "./Server";

const server = new Server(new TabbyAgent());
server.listen();
