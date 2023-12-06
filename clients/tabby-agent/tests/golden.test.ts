// This golden test requires local Tabby server to be running on port 8087.
// The server should use tabby linux image version 0.4.0, with model TabbyML/StarCoder-1B,
// cuda backend, and without any repository specified for RAG.

import { spawn } from "child_process";
import readline from "readline";
import path from "path";
import * as fs from "fs-extra";
import { expect } from "chai";

describe("agent golden test", () => {
  const agent = spawn("node", [path.join(__dirname, "../dist/cli.js")]);
  const output: any[] = [];
  readline.createInterface({ input: agent.stdout }).on("line", (line) => {
    output.push(JSON.parse(line));
  });

  const waitForResponse = (requestId: number, timeout = 1000) => {
    return new Promise<void>((resolve, reject) => {
      const start = Date.now();
      const interval = setInterval(() => {
        if (output.find((item) => item[0] === requestId)) {
          clearInterval(interval);
          resolve();
        } else if (Date.now() - start > timeout) {
          clearInterval(interval);
          reject(new Error("Timeout"));
        }
      }, 10);
    });
  };

  const createGoldenTest = async (goldenFilepath: string) => {
    const content = await fs.readFile(goldenFilepath, "utf8");
    const language = path.basename(goldenFilepath, path.extname(goldenFilepath)).replace(/^\d+-/g, "");
    const replaceStart = content.indexOf("⏩");
    const insertStart = content.indexOf("⏭");
    const insertEnd = content.indexOf("⏮");
    const replaceEnd = content.indexOf("⏪");
    const prefix = content.slice(0, replaceStart);
    const replacePrefix = content.slice(replaceStart + 1, insertStart);
    const suggestion = content.slice(insertStart + 1, insertEnd);
    const replaceSuffix = content.slice(insertEnd + 1, replaceEnd);
    const suffix = content.slice(replaceEnd + 1);
    const request = {
      filepath: goldenFilepath,
      language,
      text: prefix + replacePrefix + replaceSuffix + suffix,
      position: prefix.length + replacePrefix.length,
      manually: true,
    };
    const text = replacePrefix + suggestion;
    const expected =
      text.length == 0
        ? { choices: [] }
        : {
            choices: [
              {
                index: 0,
                text,
                replaceRange: {
                  start: prefix.length,
                  end: prefix.length + replacePrefix.length + replaceSuffix.length,
                },
              },
            ],
          };
    return { request, expected };
  };

  const config = {
    server: {
      endpoint: "http://127.0.0.1:8087",
      token: "",
      requestHeaders: {},
      requestTimeout: 30000,
    },
    completion: {
      prompt: { experimentalStripAutoClosingCharacters: false, maxPrefixLines: 20, maxSuffixLines: 20 },
      debounce: { mode: "adaptive", interval: 250 },
      timeout: 4000,
    },
    postprocess: {
      limitScope: {
        experimentalSyntax: false,
        indentation: { experimentalKeepBlockScopeWhenCompletingLine: false },
      },
      calculateReplaceRange: { experimentalSyntax: false },
    },
    logs: { level: "debug" },
    anonymousUsageTracking: { disable: true },
  };

  let requestId = 0;
  it("initialize", async () => {
    requestId++;
    const initRequest = [
      requestId,
      {
        func: "initialize",
        args: [{ config }],
      },
    ];

    agent.stdin.write(JSON.stringify(initRequest) + "\n");
    await waitForResponse(requestId);
    expect(output.shift()).to.deep.equal([0, { event: "statusChanged", status: "ready" }]);
    expect(output.shift()).to.deep.equal([0, { event: "configUpdated", config }]);
    expect(output.shift()).to.deep.equal([requestId, true]);
  });

  const goldenFiles = fs.readdirSync(path.join(__dirname, "golden")).map((file) => {
    return {
      path: path.join("golden", file),
      absolutePath: path.join(__dirname, "golden", file),
    };
  });
  goldenFiles.forEach((goldenFile) => {
    it(goldenFile.path, async () => {
      const test = await createGoldenTest(goldenFile.absolutePath);
      requestId++;
      const request = [requestId, { func: "provideCompletions", args: [test.request] }];
      agent.stdin.write(JSON.stringify(request) + "\n");
      await waitForResponse(requestId);
      const response = output.shift();
      expect(response[0]).to.equal(requestId);
      expect(response[1].choices).to.deep.equal(test.expected.choices);
    });
  });

  const badCasesFiles = fs.readdirSync(path.join(__dirname, "bad_cases")).map((file) => {
    return {
      path: path.join("bad_cases", file),
      absolutePath: path.join(__dirname, "bad_cases", file),
    };
  });
  badCasesFiles.forEach((goldenFile) => {
    it(goldenFile.path, async () => {
      const test = await createGoldenTest(goldenFile.absolutePath);
      requestId++;
      const request = [requestId, { func: "provideCompletions", args: [test.request] }];
      agent.stdin.write(JSON.stringify(request) + "\n");
      await waitForResponse(requestId);
      const response = output.shift();
      expect(response[0]).to.equal(requestId);
      expect(response[1].choices).not.to.deep.equal(test.expected.choices);
    });
  });

  it("updateConfig experimental", async () => {
    const expectedConfig = { ...config };
    {
      requestId++;
      const updateConfigRequest = [
        requestId,
        {
          func: "updateConfig",
          args: ["postprocess.limitScope.experimentalSyntax", true],
        },
      ];
      agent.stdin.write(JSON.stringify(updateConfigRequest) + "\n");
      await waitForResponse(requestId);
      expectedConfig.postprocess.limitScope.experimentalSyntax = true;
      expect(output.shift()).to.deep.equal([0, { event: "configUpdated", config: expectedConfig }]);
      expect(output.shift()).to.deep.equal([requestId, true]);
    }
    {
      requestId++;
      const updateConfigRequest = [
        requestId,
        {
          func: "updateConfig",
          args: ["postprocess.calculateReplaceRange.experimentalSyntax", true],
        },
      ];
      agent.stdin.write(JSON.stringify(updateConfigRequest) + "\n");
      await waitForResponse(requestId);
      expectedConfig.postprocess.calculateReplaceRange.experimentalSyntax = true;
      expect(output.shift()).to.deep.equal([0, { event: "configUpdated", config: expectedConfig }]);
      expect(output.shift()).to.deep.equal([requestId, true]);
    }
  });
  badCasesFiles.forEach((goldenFile) => {
    it("experimental: " + goldenFile.path, async () => {
      const test = await createGoldenTest(goldenFile.absolutePath);
      requestId++;
      const request = [requestId, { func: "provideCompletions", args: [test.request] }];
      agent.stdin.write(JSON.stringify(request) + "\n");
      await waitForResponse(requestId);
      const response = output.shift();
      expect(response[0]).to.equal(requestId);
      expect(response[1].choices).to.deep.equal(test.expected.choices);
    });
  });

  after(() => {
    agent.kill();
  });
});
