import { describe } from "mocha";
import { expect } from "chai";
import { parseInput, InlineChatParseResult } from "./util";

describe("parseInput", () => {
  it("should parse input correctly", () => {
    const input = `this is a user command`;

    const parseResult: InlineChatParseResult = {
      command: "this is a user command",
      mentions: [],
      mentionQuery: undefined,
    };
    expect(parseInput(input)).to.deep.equal(parseResult);
  });

  it("should parse input correctly", () => {
    const input = `@file1`;

    const parseResult: InlineChatParseResult = {
      command: "",
      mentions: ["file1"],
      mentionQuery: "file1",
    };
    expect(parseInput(input)).to.deep.equal(parseResult);
  });

  it("should parse input correctly", () => {
    const input = `@`;

    const parseResult: InlineChatParseResult = {
      command: "",
      mentions: [],
      mentionQuery: "",
    };
    expect(parseInput(input)).to.deep.equal(parseResult);
  });

  it("should parse input correctly", () => {
    const input = `this is a user command @`;

    const parseResult: InlineChatParseResult = {
      command: "this is a user command",
      mentions: [],
      mentionQuery: "",
    };
    expect(parseInput(input)).to.deep.equal(parseResult);
  });

  it("should parse input correctly", () => {
    const input = `this is a user command@`;

    const parseResult: InlineChatParseResult = {
      command: "this is a user command@",
      mentions: [],
      mentionQuery: undefined,
    };
    expect(parseInput(input)).to.deep.equal(parseResult);
  });

  it("should parse input correctly", () => {
    const input = ` @file1`;

    const parseResult: InlineChatParseResult = {
      command: "",
      mentions: ["file1"],
      mentionQuery: "file1",
    };
    expect(parseInput(input)).to.deep.equal(parseResult);
  });

  it("should parse input correctly", () => {
    const input = `@file1 `;

    const parseResult: InlineChatParseResult = {
      command: "",
      mentions: ["file1"],
      mentionQuery: undefined,
    };
    expect(parseInput(input)).to.deep.equal(parseResult);
  });

  it("should parse input correctly", () => {
    const input = `@file1 @file2`;
    const parseResult: InlineChatParseResult = {
      command: "",
      mentions: ["file1", "file2"],
      mentionQuery: "file2",
    };
    expect(parseInput(input)).to.deep.equal(parseResult);
  });

  it("should parse input correctly", () => {
    const input = `@file1 @file2 `;
    const parseResult: InlineChatParseResult = {
      command: "",
      mentions: ["file1", "file2"],
      mentionQuery: undefined,
    };
    expect(parseInput(input)).to.deep.equal(parseResult);
  });

  it("should parse input correctly", () => {
    const input = ` @file1 @file2 this is a user command`;
    const parseResult: InlineChatParseResult = {
      command: "this is a user command",
      mentions: ["file1", "file2"],
      mentionQuery: undefined,
    };
    expect(parseInput(input)).to.deep.equal(parseResult);
  });

  it("should parse input correctly", () => {
    const input = ` @file1 @file2 this is a user command@file3`;
    const parseResult: InlineChatParseResult = {
      command: "this is a user command@file3",
      mentions: ["file1", "file2"],
      mentionQuery: undefined,
    };
    expect(parseInput(input)).to.deep.equal(parseResult);
  });

  it("should parse input correctly", () => {
    const input = ` @file1 @file2 this is a user command @file3`;
    const parseResult: InlineChatParseResult = {
      command: "this is a user command",
      mentions: ["file1", "file2", "file3"],
      mentionQuery: "file3",
    };
    expect(parseInput(input)).to.deep.equal(parseResult);
  });

  it("should parse input correctly", () => {
    const input = ` this is a user command @file3 `;
    const parseResult: InlineChatParseResult = {
      command: "this is a user command",
      mentions: ["file3"],
      mentionQuery: undefined,
    };
    expect(parseInput(input)).to.deep.equal(parseResult);
  });

  it("should parse input correctly", () => {
    const input = ` this is a @file3  user command `;
    const parseResult: InlineChatParseResult = {
      command: "this is a   user command",
      mentions: ["file3"],
      mentionQuery: undefined,
    };
    expect(parseInput(input)).to.deep.equal(parseResult);
  });
});
