import fixSpellingAndGrammarPrompt from "./prompts/fix-spelling-and-grammar.md";
import generateCommitMessagePrompt from "./prompts/generate-commit-message.md";
import generateDocsPrompt from "./prompts/generate-docs.md";
import editCommandReplacePrompt from "./prompts/edit-command-replace.md";
import editCommandInsertPrompt from "./prompts/edit-command-insert.md";

export type AgentConfig = {
  server: {
    endpoint: string;
    token: string;
    requestHeaders: Record<string, string | number | boolean | null | undefined>;
    requestTimeout: number;
  };
  proxy: {
    authorization: string;
    url: string;
    noProxy: string | string[];
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
  chat: {
    edit: {
      documentMaxChars: number;
      commandMaxChars: number;
      responseDocumentTag: string[];
      responseCommentTag: string[] | undefined;
      promptTemplate: {
        replace: string;
        insert: string;
      };
      presetCommands: Record<
        string,
        {
          label: string;
          filters: Record<string, string>;
          kind: "replace" | "insert";
          promptTemplate: string;
        }
      >;
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
  proxy: {
    authorization: "",
    url: "",
    noProxy: "",
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
  chat: {
    edit: {
      documentMaxChars: 3000,
      commandMaxChars: 200,
      responseDocumentTag: ["<GENERATEDCODE>", "</GENERATEDCODE>"],
      responseCommentTag: undefined,
      promptTemplate: {
        replace: editCommandReplacePrompt,
        insert: editCommandInsertPrompt,
      },
      presetCommands: {
        "/doc": {
          label: "Generate Docs",
          filters: { languageIdNotIn: "plaintext,markdown" },
          kind: "replace",
          promptTemplate: generateDocsPrompt,
        },
        "/fix": {
          label: "Fix spelling and grammar errors",
          filters: { languageIdIn: "plaintext,markdown" },
          kind: "replace",
          promptTemplate: fixSpellingAndGrammarPrompt,
        },
      },
    },
    generateCommitMessage: {
      maxDiffLength: 3600,
      promptTemplate: generateCommitMessagePrompt,
      responseMatcher:
        /(?<=(["'`]+)?\s*)(feat|fix|docs|refactor|style|test|build|ci|chore)(\(\w+\))?:.+(?=\s*\1)/gi.toString(),
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
