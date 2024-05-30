export type AgentConfig = {
  server: {
    endpoint: string;
    token: string;
    requestHeaders: Record<string, string | number | boolean | null | undefined>;
    requestTimeout: number;
  };
  completion: {
    prompt: {
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
    solution: {
      // The max number of unique choices to be fetched before stopping
      maxItems: number;
      // The max number of attempts to fetch choices before stopping
      maxTries: number;
      // The temperature for fetching the second and subsequent choices
      temperature: number;
    };
  };
  postprocess: {
    limitScope: any;
    calculateReplaceRange: any;
  };
  experimentalChat: {
    edit: {
      documentMaxChars: number;
      commandMaxChars: number;
      promptTemplate: string;
      responseSplitter: string;
      responseSplitterIncrement: string;
    };
    generateCommitMessage: {
      maxDiffLength: number;
      promptTemplate: string;
      responseMatcher: string;
    };
  };
  logs: {
    // Controls the level of the logger written to the `~/.tabby-client/agent/logs/`
    level: "silent" | "error" | "info" | "debug" | "trace";
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
      maxPrefixLines: 20,
      maxSuffixLines: 20,
      fillDeclarations: {
        enabled: true,
        maxSnippets: 5,
        maxCharsPerSnippet: 500,
      },
      collectSnippetsFromRecentChangedFiles: {
        enabled: true,
        maxSnippets: 3,
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
    solution: {
      maxItems: 3,
      maxTries: 6,
      temperature: 0.8,
    },
  },
  postprocess: {
    limitScope: {},
    calculateReplaceRange: {},
  },
  experimentalChat: {
    edit: {
      documentMaxChars: 3600,
      commandMaxChars: 200,
      promptTemplate:
        'Edit the given document according to the given command. You should use the same language with the given document if not specified. You should keep the leading indentation or empty lines if not formatting the document. You must reply the edited document quoted by 6 backticks, in the following format: \n``````\nyour edited document \n``````\n\n After the edited document, you can optionally add a comment to briefly describe your changes. \n\n\nThe command:  \n"{{command}}" \n\nThe document: \n``````{{languageId}}\n{{document}} \n``````\n',
      responseSplitter: "```",
      responseSplitterIncrement: "`",
    },
    generateCommitMessage: {
      maxDiffLength: 3600,
      promptTemplate:
        "Generate a commit message based on the given diff. \nYou should only reply with the commit message, and the commit message should be in the following format: <type>(<scope>): <description> \nexamples: \n * feat(chat): add support for chat. \n * fix(ui): fix homepage links. \nThe diff is: \n\n```diff \n{{diff}} \n``` \n",
      responseMatcher: /(?<=^\s*(the commit message.*:\s+)|(`{3}|["'`])\s*)[^"'`\s].*(?=\s*\2\s*$)/gi.toString(),
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
