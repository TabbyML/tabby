import path from "path";
import fs from "fs-extra";
import { v4 as uuid } from "uuid";
import toml from "toml";
import glob from "glob";
import { expect } from "chai";
import { deepmerge } from "deepmerge-ts";
import { ConfigData } from "../../config/type";
import { defaultConfigData } from "../../config/default";
import { CompletionItem } from "../solution";
import { CompletionContext } from "../contexts";
import { preCacheProcess, postCacheProcess } from "./index";

type PostprocessConfig = ConfigData["postprocess"];

type DocumentContext = {
  prefix: string;
  replacePrefix: string;
  completion: string;
  replaceSuffix: string;
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
    replacePrefix: text.slice(replaceStart + 1, insertStart),
    completion: text.slice(insertStart + 1, insertEnd),
    replaceSuffix: text.slice(insertEnd + 1, replaceEnd),
    suffix: text.slice(replaceEnd + 1),
  };
}

function getDoc(context: DocumentContext): string {
  return context.prefix + context.replacePrefix + context.replaceSuffix + context.suffix;
}

function getPosition(context: DocumentContext): number {
  return context.prefix.length + context.replacePrefix.length;
}

function getCompletionFullText(context: DocumentContext): string {
  return context.replacePrefix + context.completion;
}

describe("postprocess golden test", () => {
  const postprocess = async (item: CompletionItem, config: PostprocessConfig): Promise<CompletionItem> => {
    let processed = await preCacheProcess([item], config);
    processed = await postCacheProcess(processed, config);
    return processed[0]!;
  };

  const files = glob.sync(path.join(__dirname, "golden/**/*.toml"));
  files.sort().forEach((file) => {
    const fileContent = fs.readFileSync(file, "utf8");
    const testCase = toml.parse(fileContent);
    it(testCase["description"] ?? file, async () => {
      const config = deepmerge(defaultConfigData["postprocess"], testCase["config"] ?? {}) as PostprocessConfig;
      const docContext = parseDocContext(testCase["context"]?.["text"] ?? "");
      const context = new CompletionContext({
        filepath: testCase["context"]?.["filepath"] ?? uuid(),
        language: testCase["context"]?.["language"] ?? "plaintext",
        text: getDoc(docContext),
        position: getPosition(docContext),
        indentation: testCase["context"]?.["indentation"],
      });
      const completionItem = new CompletionItem(
        context,
        getCompletionFullText(docContext),
        docContext.replacePrefix,
        docContext.replaceSuffix,
      );
      const unchanged = completionItem;
      const output = await postprocess(completionItem, config);

      const checkExpected = (expected: CompletionItem) => {
        if (testCase["expected"]?.["notEqual"]) {
          expect(output).to.not.deep.equal(expected);
        } else {
          expect(output).to.deep.equal(expected);
        }
      };

      if (testCase["expected"]?.["unchanged"]) {
        checkExpected(unchanged);
      } else if (testCase["expected"]?.["discard"]) {
        const expected = CompletionItem.createBlankItem(context);
        checkExpected(expected);
      } else {
        const expectedContext = parseDocContext(testCase["expected"]?.["text"] ?? "");
        const expected = new CompletionItem(
          context,
          getCompletionFullText(expectedContext),
          expectedContext.replacePrefix,
          expectedContext.replaceSuffix,
        );
        checkExpected(expected);
      }
    });
  });
});
