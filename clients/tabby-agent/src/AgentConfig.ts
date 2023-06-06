export type AgentConfig = {
  server?: {
    endpoint?: string;
  };
  logs?: {
    level?: "debug" | "error" | "silent";
  };
  anonymousUsageTracking?: {
    disable?: boolean;
  };
};

export const defaultAgentConfig: AgentConfig = {
  server: {
    endpoint: "http://localhost:8080",
  },
  logs: {
    level: "error",
  },
  anonymousUsageTracking: {
    disable: false,
  },
};
