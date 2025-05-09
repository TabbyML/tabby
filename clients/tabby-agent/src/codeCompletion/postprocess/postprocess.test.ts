import { TextDocument } from "vscode-languageserver-textdocument";
import path from "path";
import fs from "fs-extra";
import { v4 as uuid } from "uuid";
import toml from "toml";
import glob from "glob";
import { expect } from "chai";
import { deepmerge } from "deepmerge-ts";
import { ConfigData } from "../../config/type";
import { defaultConfigData } from "../../config/default";
import { CompletionResultItem, emptyCompletionResultItem } from "../solution";
import { buildCompletionContext, CompletionContext, CompletionExtraContexts } from "../contexts";
import { preCacheProcess, postCacheProcess } from "./index";

type PostprocessConfig = ConfigData["postprocess"];

type DocumentContext = {
  prefix: string;
  completion: string;
  replaceSuffix: string;
  suffix: string;
};

function parseDocContext(text: string): DocumentContext {
  const insertStart = text.indexOf("├");
  const insertEnd = text.lastIndexOf("┤");
  let replaceEnd = text.lastIndexOf("╣");
  if (replaceEnd < 0) {
    replaceEnd = insertEnd;
  }
  return {
    prefix: text.slice(0, insertStart),
    completion: text.slice(insertStart + 1, insertEnd),
    replaceSuffix: text.slice(insertEnd + 1, replaceEnd),
    suffix: text.slice(replaceEnd + 1),
  };
}

function getDoc(context: DocumentContext): string {
  return context.prefix + context.replaceSuffix + context.suffix;
}

function getPosition(context: DocumentContext): number {
  return context.prefix.length;
}

function getCompletionText(context: DocumentContext): string {
  return context.completion;
}

describe("postprocess golden test", () => {
  const postprocess = async (
    item: CompletionResultItem,
    context: CompletionContext,
    extraContext: CompletionExtraContexts,
    config: PostprocessConfig,
  ): Promise<CompletionResultItem> => {
    let processed = await preCacheProcess([item], context, extraContext, config);
    processed = await postCacheProcess(processed, context, extraContext, config);
    return processed[0]!;
  };

  const files = glob.sync(path.join(__dirname, "golden/**/*.toml"));
  files.sort().forEach((file) => {
    const fileContent = fs.readFileSync(file, "utf8");
    const testCase = toml.parse(fileContent);
    it(testCase["description"] ?? file, async () => {
      const config = deepmerge(defaultConfigData["postprocess"], testCase["config"] ?? {}) as PostprocessConfig;
      const docContext = parseDocContext(testCase["context"]?.["text"] ?? "");
      const textDocument = TextDocument.create(
        testCase["context"]?.["filepath"] ?? uuid(),
        testCase["context"]?.["language"] ?? "plaintext",
        0,
        getDoc(docContext),
      );
      const context = buildCompletionContext(textDocument, textDocument.positionAt(getPosition(docContext)));
      const completionItem = new CompletionResultItem(getCompletionText(docContext));
      const unchanged = completionItem;
      const output = await postprocess(
        completionItem,
        context,
        { editorOptions: { indentation: testCase["context"]?.["indentation"] } },
        config,
      );

      const checkExpected = (expected: CompletionResultItem) => {
        if (testCase["expected"]?.["notEqual"]) {
          expect(output).to.not.deep.equal(expected);
        } else {
          expect(output).to.deep.equal(expected);
        }
      };

      if (testCase["expected"]?.["unchanged"]) {
        checkExpected(unchanged);
      } else if (testCase["expected"]?.["discard"]) {
        const expected = emptyCompletionResultItem;
        checkExpected(expected);
      } else {
        const expectedContext = parseDocContext(testCase["expected"]?.["text"] ?? "");
        const expected = new CompletionResultItem(getCompletionText(expectedContext));
        checkExpected(expected);
      }
    });
  });
});
