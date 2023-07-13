#!/bin/env node

import { TabbyAgent } from "./TabbyAgent";
import { StdIO } from "./StdIO";

const stdio = new StdIO();
TabbyAgent.create().then((agent) => {
  stdio.bind(agent);
  stdio.listen();
});
