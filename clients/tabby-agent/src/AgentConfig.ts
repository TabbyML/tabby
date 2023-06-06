export type AgentConfig = {
  server?: {
    endpoint?: string;
  };
  logs?: {
    level?: "debug" | "error" | "silent";
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
    level: "silent",
  },
  analytics: {
    enabled: true,
  },
};
