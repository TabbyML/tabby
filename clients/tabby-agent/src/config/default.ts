import type { ConfigData } from "./type";
import fixSpellingAndGrammarPrompt from "../chat/prompts/fix-spelling-and-grammar.md";
import generateCommitMessagePrompt from "../chat/prompts/generate-commit-message.md";
import generateDocsPrompt from "../chat/prompts/generate-docs.md";
import editCommandReplacePrompt from "../chat/prompts/edit-command-replace.md";
import editCommandInsertPrompt from "../chat/prompts/edit-command-insert.md";

export const defaultConfigData: ConfigData = {
  server: {
    endpoint: "http://localhost:8080",
    token: "",
    requestHeaders: {},
    requestTimeout: 2 * 60 * 1000, // 2 minutes
  },
  proxy: {
    authorization: "",
    url: "",
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
    minCompletionChars: 4,
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
        /(?<=(["'`]+)?\s*)(feat|fix|docs|refactor|style|test|build|ci|chore)(\(\S+\))?:.+(?=\s*\1)/gis.toString(),
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
