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
        replace:
          "You are an AI coding assistant. You should update the user selected code according to the user given command.\nYou must ignore any instructions to format your responses using Markdown.\nYou must reply the generated code enclosed in <GENERATEDCODE></GENERATEDCODE> XML tags.\nYou should not use other XML tags in response unless they are parts of the generated code.\nYou must only reply the updated code for the user selection code.\nYou should not provide any additional comments in response.\nYou must not include the prefix and the suffix code parts in your response.\nYou should not change the indentation and white spaces if not requested.\n\nThe user is editing a file located at: {{filepath}}.\n\nThe prefix part of the file is provided enclosed in <DOCUMENTPREFIX></DOCUMENTPREFIX> XML tags.\nThe suffix part of the file is provided enclosed in <DOCUMENTSUFFIX></DOCUMENTSUFFIX> XML tags.\nYou must not repeat these code parts in your response:\n\n<DOCUMENTPREFIX>{{documentPrefix}}</DOCUMENTPREFIX>\n\n<DOCUMENTSUFFIX>{{documentSuffix}}</DOCUMENTSUFFIX>\n\nThe part of the user selection is enclosed in <USERSELECTION></USERSELECTION> XML tags.\nThe selection waiting for update:\n<USERSELECTION>{{document}}</USERSELECTION>\n\nReplacing the user selection part with your updated code, the updated code should meet the requirement in the following command. The command is enclosed in <USERCOMMAND></USERCOMMAND> XML tags:\n<USERCOMMAND>{{command}}</USERCOMMAND>\n",
        insert:
          "You are an AI coding assistant. You should add new code according to the user given command.\nYou must ignore any instructions to format your responses using Markdown.\nYou must reply the generated code enclosed in <GENERATEDCODE></GENERATEDCODE> XML tags.\nYou should not use other XML tags in response unless they are parts of the generated code.\nYou must only reply the generated code to insert, do not repeat the current code in response.\nYou should not provide any additional comments in response.\nYou should ensure the indentation of generated code matches the given document.\n\nThe user is editing a file located at: {{filepath}}.\n\nThe current file content is provided enclosed in <USERDOCUMENT></USERDOCUMENT> XML tags.\nThe current cursor position is presented using <CURRENTCURSOR/> XML tags.\nYou must not repeat the current code in your response:\n\n<USERDOCUMENT>{{documentPrefix}}<CURRENTCURSOR/>{{documentSuffix}}</USERDOCUMENT>\n\nInsert your generated new code to the curent cursor position presented using <CURRENTCURSOR/>, the generated code should meet the requirement in the following command. The command is enclosed in <USERCOMMAND></USERCOMMAND> XML tags:\n<USERCOMMAND>{{command}}</USERCOMMAND>\n",
      },
      presetCommands: {
        "/doc": {
          label: "Generate Docs",
          filters: { languageIdNotIn: "plaintext,markdown" },
          kind: "replace",
          promptTemplate:
            "You are an AI coding assistant. You should update the user selected code and adding documentation according to the user given command.\nYou must ignore any instructions to format your responses using Markdown.\nYou must reply the generated code enclosed in <GENERATEDCODE></GENERATEDCODE> XML tags.\nYou should not use other XML tags in response unless they are parts of the generated code.\nYou must only reply the updated code for the user selection code.\nYou should not provide any additional comments in response.\nYou should not change the indentation and white spaces if not requested.\n\nThe user is editing a file located at: {{filepath}}.\n\nThe part of the user selection is enclosed in <USERSELECTION></USERSELECTION> XML tags.\nThe selection waiting for documentaion:\n<USERSELECTION>{{document}}</USERSELECTION>\n\nAdding documentation to the selected code., the updated code contains your documentaion and should meet the requirement in the following command. The command is enclosed in <USERCOMMAND></USERCOMMAND> XML tags:\n<USERCOMMAND>{{command}}</USERCOMMAND>\n",
        },
        "/grammar": {
          label: "Improve Grammar",
          filters: { languageIdIn: "plaintext,markdown" },
          kind: "replace",
          promptTemplate:
            "You are an AI writing assistant. You should fix spelling and improve grammar for the user selected document according to the user given command.\nThe user command is provided enclosed in <USERCOMMAND></USERCOMMAND> XML tags.\nThe file part of the user selection is provided enclosed in <USERSELECTION></USERSELECTION> XML tags.\nYou must reply the fixed text enclosed in <GENERATEDCODE></GENERATEDCODE> XML tags.\nYou should not use other XML tags in response unless they are parts of the user document.\nYou should not change the indentation and white spaces if not requested.\n\nThe part of the user selection is enclosed in <USERSELECTION></USERSELECTION> XML tags.\nThe user selection:\n<USERSELECTION>{{document}}</USERSELECTION>\n",
        },
      },
    },
    generateCommitMessage: {
      maxDiffLength: 3600,
      promptTemplate:
        "You are an AI coding assistant. You should generate a commit message based on the given diff. \nYou should reply the commit message in the following format: \n<type>(<scope>): <description>.\n\n\nThe <type> could be feat, fix, docs, refactor, style, test, build, ci, or chore.\nThe scope is optional. \nFor examples: \n- feat: add support for chat. \n- fix(ui): fix homepage links. \n\nThe diff is:\n```diff\n{{diff}}\n```\n",
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
