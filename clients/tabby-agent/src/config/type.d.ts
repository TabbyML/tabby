export type ConfigData = {
  server: {
    endpoint: string;
    token: string;
    requestHeaders: Record<string, string | number | boolean | null | undefined>;
    requestTimeout: number;
  };
  proxy: {
    authorization: string;
    url: string;
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
    minCompletionChars: number;
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

export type PartialConfigData = RecursivePartial<ConfigData>;
