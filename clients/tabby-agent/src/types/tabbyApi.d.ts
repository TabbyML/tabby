export interface paths {
  "/v1/chat/completions": {
    post: operations["chat_completions"];
  };
  "/v1/completions": {
    post: operations["completion"];
  };
  "/v1/events": {
    post: operations["event"];
  };
  "/v1/health": {
    get: operations["health"];
    // back compatible for Tabby server 0.2.x and earlier
    post: operations["health"];
  };
  // back compatible for Tabby server 0.10.x and earlier
  "/v1beta/chat/completions": {
    post: operations["chat_completions"];
  };
  "/v1beta/search": {
    get: operations["search"];
  };
  "/v1beta/server_setting": {
    get: operations["config"];
  };
}

export type webhooks = Record<string, never>;

export interface components {
  schemas: {
    ChatCompletionChoice: {
      index: number;
      logprobs?: string | null;
      finish_reason?: string | null;
      delta: components["schemas"]["ChatCompletionDelta"];
    };
    ChatCompletionChunk: {
      id: string;
      /** Format: int64 */
      created: number;
      system_fingerprint: string;
      object: string;
      model: string;
      choices: components["schemas"]["ChatCompletionChoice"][];
    };
    ChatCompletionDelta: {
      content: string;
    };
    /**
     * @example {
     *   "messages": [
     *     {
     *       "content": "What is tail recursion?",
     *       "role": "user"
     *     },
     *     {
     *       "content": "It's a kind of optimization in compiler?",
     *       "role": "assistant"
     *     },
     *     {
     *       "content": "Could you share more details?",
     *       "role": "user"
     *     }
     *   ]
     * }
     */
    ChatCompletionRequest: {
      messages: components["schemas"]["Message"][];
      /** Format: float */
      temperature?: number | null;
      /** Format: int64 */
      seed?: number | null;
    };
    Choice: {
      /** Format: int32 */
      index: number;
      text: string;
    };
    /**
     * @example {
     *   "language": "python",
     *   "segments": {
     *     "prefix": "def fib(n):\n    ",
     *     "suffix": "\n        return fib(n - 1) + fib(n - 2)"
     *   }
     * }
     */
    CompletionRequest: {
      /**
       * @description Language identifier, full list is maintained at
       * https://code.visualstudio.com/docs/languages/identifiers
       * @example python
       */
      language?: string | null;
      segments?: components["schemas"]["Segments"] | null;
      /**
       * @description A unique identifier representing your end-user, which can help Tabby to monitor & generating
       * reports.
       */
      user?: string | null;
      debug_options?: components["schemas"]["DebugOptions"] | null;
      /**
       * Format: float
       * @description The temperature parameter for the model, used to tune variance and "creativity" of the model output
       */
      temperature?: number | null;
      /**
       * Format: int64
       * @description The seed used for randomly selecting tokens
       */
      seed?: number | null;
    };
    /**
     * @example {
     *   "choices": [
     *     {
     *       "index": 0,
     *       "text": "string"
     *     }
     *   ],
     *   "id": "string"
     * }
     */
    CompletionResponse: {
      id: string;
      choices: components["schemas"]["Choice"][];
      debug_data?: components["schemas"]["DebugData"] | null;
    };
    DebugData: {
      snippets?: components["schemas"]["Snippet"][] | null;
      prompt?: string | null;
    };
    DebugOptions: {
      /**
       * @description When `raw_prompt` is specified, it will be passed directly to the inference engine for completion. `segments` field in `CompletionRequest` will be ignored.
       *
       * This is useful for certain requests that aim to test the tabby's e2e quality.
       */
      raw_prompt?: string | null;
      /** @description When true, returns `snippets` in `debug_data`. */
      return_snippets?: boolean;
      /** @description When true, returns `prompt` in `debug_data`. */
      return_prompt?: boolean;
      /** @description When true, disable retrieval augmented code completion. */
      disable_retrieval_augmented_code_completion?: boolean;
    };
    /** @description A snippet of declaration code that is relevant to the current completion request. */
    Declaration: {
      /**
       * @description Filepath of the file where the snippet is from.
       * - When the file belongs to the same workspace as the current file,
       * this is a relative filepath, use the same rule as [Segments::filepath].
       * - When the file located outside the workspace, such as in a dependency package,
       * this is a file URI with an absolute filepath.
       */
      filepath: string;
      /** @description Body of the snippet. */
      body: string;
    };
    HealthState: {
      model?: string | null;
      chat_model?: string | null;
      chat_device?: string | null;
      device: string;
      arch: string;
      cpu_info: string;
      cpu_count: number;
      cuda_devices: string[];
      version: components["schemas"]["Version"];
      webserver?: boolean | null;
    };
    Hit: {
      /** Format: float */
      score: number;
      doc: components["schemas"]["HitDocument"];
      /** Format: int32 */
      id: number;
    };
    HitDocument: {
      body: string;
      filepath: string;
      git_url: string;
      language: string;
    };
    LogEventRequest: {
      /**
       * @description Event type, should be `view`, `select` or `dismiss`.
       * @example view
       */
      type: string;
      completion_id: string;
      /** Format: int32 */
      choice_index: number;
      view_id?: string | null;
      /** Format: int32 */
      elapsed?: number | null;
    };
    Message: {
      role: string;
      content: string;
    };
    SearchResponse: {
      num_hits: number;
      hits: components["schemas"]["Hit"][];
    };
    Segments: {
      /** @description Content that appears before the cursor in the editor window. */
      prefix: string;
      /** @description Content that appears after the cursor in the editor window. */
      suffix?: string | null;
      /**
       * @description The relative path of the file that is being edited.
       * - When [Segments::git_url] is set, this is the path of the file in the git repository.
       * - When [Segments::git_url] is empty, this is the path of the file in the workspace.
       */
      filepath?: string | null;
      /**
       * @description The remote URL of the current git repository.
       * Leave this empty if the file is not in a git repository,
       * or the git repository does not have a remote URL.
       */
      git_url?: string | null;
      /**
       * @description The relevant declaration code snippets provided by the editor's LSP,
       * contain declarations of symbols extracted from [Segments::prefix].
       */
      declarations?: components["schemas"]["Declaration"][] | null;
      /**
       * @description The relevant code snippets extracted from recently edited files.
       * These snippets are selected from candidates found within code chunks
       * based on the edited location.
       * The current editing file is excluded from the search candidates.
       *
       * When provided alongside [Segments::declarations], the snippets have
       * already been deduplicated to ensure no duplication with entries
       * in [Segments::declarations].
       *
       * Sorted in descending order of [Snippet::score].
       */
      relevant_snippets_from_changed_files?: components["schemas"]["Snippet"][] | null;
      /** @description Clipboard content when requesting code completion. */
      clipboard?: string | null;
    };
    ServerSetting: {
      disable_client_side_telemetry: boolean;
    };
    Snippet: {
      filepath: string;
      body: string;
      /** Format: float */
      score: number;
    };
    Version: {
      build_date: string;
      build_timestamp: string;
      git_sha: string;
      git_describe: string;
    };
  };
  responses: never;
  parameters: never;
  requestBodies: never;
  headers: never;
  pathItems: never;
}

export type $defs = Record<string, never>;

export type external = Record<string, never>;

export interface operations {
  chat_completions: {
    requestBody: {
      content: {
        "application/json": components["schemas"]["ChatCompletionRequest"];
      };
    };
    responses: {
      /** @description Success */
      200: {
        content: {
          "text/event-stream": components["schemas"]["ChatCompletionChunk"];
        };
      };
      /** @description When chat model is not specified, the endpoint returns 405 Method Not Allowed */
      405: {
        content: never;
      };
      /** @description When the prompt is malformed, the endpoint returns 422 Unprocessable Entity */
      422: {
        content: never;
      };
    };
  };
  completion: {
    requestBody: {
      content: {
        "application/json": components["schemas"]["CompletionRequest"];
      };
    };
    responses: {
      /** @description Success */
      200: {
        content: {
          "application/json": components["schemas"]["CompletionResponse"];
        };
      };
      /** @description Bad Request */
      400: {
        content: never;
      };
    };
  };
  event: {
    parameters: {
      query: {
        select_kind?: string | null;
      };
    };
    requestBody: {
      content: {
        "application/json": components["schemas"]["LogEventRequest"];
      };
    };
    responses: {
      /** @description Success */
      200: {
        content: never;
      };
      /** @description Bad Request */
      400: {
        content: never;
      };
    };
  };
  health: {
    responses: {
      /** @description Success */
      200: {
        content: {
          "application/json": components["schemas"]["HealthState"];
        };
      };
    };
  };
  search: {
    parameters: {
      query: {
        q: string;
        limit?: number | null;
        offset?: number | null;
      };
    };
    responses: {
      /** @description Success */
      200: {
        content: {
          "application/json": components["schemas"]["SearchResponse"];
        };
      };
      /** @description When code search is not enabled, the endpoint will returns 501 Not Implemented */
      501: {
        content: never;
      };
    };
  };
  config: {
    responses: {
      /** @description Success */
      200: {
        content: {
          "application/json": components["schemas"]["ServerSetting"];
        };
      };
    };
  };
}
