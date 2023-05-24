import { TabbyAgent } from "./TabbyAgent";
import { StdIO } from "./StdIO";

const agent = new TabbyAgent();
const stdio = new StdIO();
stdio.bind(agent);
stdio.listen();
