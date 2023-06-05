export type AgentConfig = {
  server?: {
    endpoint?: string;
  };
  logs?: {
    console?: {
      level?: "debug" | "error" | "off";
    };
    file?: {
      level?: "debug" | "error" | "off";
      path?: string;
    };
  };
  analytics?: {
    enabled?: boolean;
  };
};

export const defaultAgentConfig: AgentConfig = {
  server: {
    endpoint: "http://localhost:8080",
  },
  logs: {
    console: {
      level: "off",
    },
    file: {
      level: "off",
      path: "~/.tabby/agent-logs",
    },
  },
  analytics: {
    enabled: true,
  },
};
