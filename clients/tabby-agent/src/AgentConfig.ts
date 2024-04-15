export type AgentConfig = {
  server: {
    endpoint: string;
    token: string;
    requestHeaders: Record<string, string | number | boolean | null | undefined>;
    requestTimeout: number;
  };
  completion: {
    prompt: {
      experimentalStripAutoClosingCharacters: boolean;
      maxPrefixLines: number;
      maxSuffixLines: number;
      fillDeclarations: {
        enabled: boolean;
        // max number of declaration snippets
        maxSnippets: number;
        // max number of characters per snippet
        maxCharsPerSnippet: number;
      };
      collectSnippetsFromRecentChangedFiles: {
        enabled: boolean;
        // max number of snippets
        maxSnippets: number;
        indexing: {
          // Interval in ms for indexing worker to check pending task
          checkingChangesInterval: number;
          // Debouncing interval in ms for sending changes to indexing task
          changesDebouncingInterval: number;

          // Determine the crop window at changed location for indexing
          // Line before changed location
          prefixLines: number;
          // Line after changed location
          suffixLines: number;

          // Max number of chunks in memory
          maxChunks: number;
          // chars per code chunk
          chunkSize: number;
          // overlap lines between neighbor chunks
          overlapLines: number;
        };
      };
      clipboard: {
        minChars: number;
        maxChars: number;
      };
    };
    debounce: {
      mode: "adaptive" | "fixed";
      interval: number;
    };
  };
  postprocess: {
    limitScope: {
      // Prefer to use syntax parser than indentation
      experimentalSyntax: boolean;
      indentation: {
        // When completion is continuing the current line, limit the scope to:
        // false(default): the line scope, meaning use the next indent level as the limit.
        // true: the block scope, meaning use the current indent level as the limit.
        experimentalKeepBlockScopeWhenCompletingLine: boolean;
      };
    };
    calculateReplaceRange: {
      // Prefer to use syntax parser than bracket stack
      experimentalSyntax: boolean;
    };
  };
  logs: {
    level: "debug" | "error" | "silent";
  };
  tls: {
    // `bundled`, `system`, or a string point to cert file
    caCerts: string;
  };
  anonymousUsageTracking: {
    disable: boolean;
  };
};

type RecursivePartial<T> = {
  [P in keyof T]?: T[P] extends (infer U)[]
    ? RecursivePartial<U>[]
    : T[P] extends object | undefined
      ? RecursivePartial<T[P]>
      : T[P];
};

export type PartialAgentConfig = RecursivePartial<AgentConfig>;

export const defaultAgentConfig: AgentConfig = {
  server: {
    endpoint: "http://localhost:8080",
    token: "",
    requestHeaders: {},
    requestTimeout: 2 * 60 * 1000, // 2 minutes
  },
  completion: {
    prompt: {
      experimentalStripAutoClosingCharacters: false,
      maxPrefixLines: 20,
      maxSuffixLines: 20,
      fillDeclarations: {
        enabled: true,
        maxSnippets: 5,
        maxCharsPerSnippet: 500,
      },
      collectSnippetsFromRecentChangedFiles: {
        enabled: false,
        maxSnippets: 5,
        indexing: {
          checkingChangesInterval: 500,
          changesDebouncingInterval: 1000,
          prefixLines: 20,
          suffixLines: 20,
          maxChunks: 100,
          chunkSize: 500,
          overlapLines: 1,
        },
      },
      clipboard: {
        minChars: 3,
        maxChars: 2000,
      },
    },
    debounce: {
      mode: "adaptive",
      interval: 250, // ms
    },
  },
  postprocess: {
    limitScope: {
      experimentalSyntax: false,
      indentation: {
        experimentalKeepBlockScopeWhenCompletingLine: false,
      },
    },
    calculateReplaceRange: {
      experimentalSyntax: false,
    },
  },
  logs: {
    level: "silent",
  },
  tls: {
    caCerts: "system",
  },
  anonymousUsageTracking: {
    disable: false,
  },
};
