import path from "path";
import fs from "fs-extra";
import { v4 as uuid } from "uuid";
import toml from "toml";
import glob from "glob";
import { expect } from "chai";
import { deepmerge } from "deepmerge-ts";
import { AgentConfig, defaultAgentConfig } from "../AgentConfig";
import { CompletionContext, CompletionResponse } from "../CompletionContext";
import { preCacheProcess, postCacheProcess, calculateReplaceRange } from ".";

type PostprocessConfig = AgentConfig["postprocess"];

type DocumentContext = {
  prefix: string;
  prefixReplaceRange: string;
  completion: string;
  suffixReplaceRange: string;
  suffix: string;
};

function parseDocContext(text: string): DocumentContext {
  const insertStart = text.indexOf("├");
  const insertEnd = text.lastIndexOf("┤");
  let replaceStart = text.indexOf("╠");
  if (replaceStart < 0) {
    replaceStart = insertStart;
  }
  let replaceEnd = text.lastIndexOf("╣");
  if (replaceEnd < 0) {
    replaceEnd = insertEnd;
  }
  return {
    prefix: text.slice(0, replaceStart),
    prefixReplaceRange: text.slice(replaceStart + 1, insertStart),
    completion: text.slice(insertStart + 1, insertEnd),
    suffixReplaceRange: text.slice(insertEnd + 1, replaceEnd),
    suffix: text.slice(replaceEnd + 1),
  };
}

function getDoc(context: DocumentContext): string {
  return context.prefix + context.prefixReplaceRange + context.suffixReplaceRange + context.suffix;
}

function getPosition(context: DocumentContext): number {
  return context.prefix.length + context.prefixReplaceRange.length;
}

function getCompletion(context: DocumentContext): string {
  return context.prefixReplaceRange + context.completion;
}

function getReplaceRange(context: DocumentContext) {
  return {
    start: context.prefix.length,
    end: context.prefix.length + context.prefixReplaceRange.length + context.suffixReplaceRange.length,
  };
}

function buildChoices(context: DocumentContext) {
  const text = getCompletion(context);
  if (text.length === 0) {
    return [];
  }
  return [
    {
      index: 0,
      text,
      replaceRange: getReplaceRange(context),
    },
  ];
}

describe("postprocess golden test", () => {
  const postprocess = async (context: CompletionContext, config: PostprocessConfig, response: CompletionResponse) => {
    let processed = await preCacheProcess(context, config, response);
    processed = await postCacheProcess(context, config, processed);
    processed = await calculateReplaceRange(context, config, processed);
    return processed;
  };

  const files = glob.sync(path.join(__dirname, "golden/**/*.toml"));
  files.sort().forEach((file) => {
    const fileContent = fs.readFileSync(file, "utf8");
    const testCase = toml.parse(fileContent);
    it(testCase["description"] ?? file, async () => {
      const config = deepmerge(defaultAgentConfig["postprocess"], testCase["config"] ?? {}) as PostprocessConfig;
      const docContext = parseDocContext(testCase["context"]?.["text"] ?? "");
      const completionContext = new CompletionContext({
        filepath: testCase["context"]?.["filepath"] ?? uuid(),
        language: testCase["context"]?.["language"] ?? "plaintext",
        text: getDoc(docContext),
        position: getPosition(docContext),
        indentation: testCase["context"]?.["indentation"],
      });
      const completionId = "test-" + uuid();
      const completionResponse = {
        id: completionId,
        choices: buildChoices(docContext),
      };
      const unchanged: CompletionResponse = JSON.parse(JSON.stringify(completionResponse));
      const output = await postprocess(completionContext, config, completionResponse);

      const checkExpected = (expected: CompletionResponse) => {
        if (testCase["expected"]?.["notEqual"]) {
          expect(output).to.not.deep.equal(expected);
        } else {
          expect(output).to.deep.equal(expected);
        }
      };

      if (testCase["expected"]?.["unchanged"]) {
        checkExpected(unchanged);
      } else if (testCase["expected"]?.["discard"]) {
        const expected = {
          id: completionId,
          choices: [],
        };
        checkExpected(expected);
      } else {
        const expectedContext = parseDocContext(testCase["expected"]?.["text"] ?? "");
        const expected = {
          id: completionId,
          choices: buildChoices(expectedContext),
        };
        checkExpected(expected);
      }
    });
  });
});
