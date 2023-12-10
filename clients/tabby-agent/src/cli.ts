#!/bin/env node

import { TabbyAgent } from "./TabbyAgent";
import { StdIO } from "./StdIO";

const stdio = new StdIO();
const agent = new TabbyAgent();
stdio.bind(agent);
stdio.listen();
