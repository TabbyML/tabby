import { describe } from "mocha";
import { expect } from "chai";
import { parseUserCommand, InlineEditParseResult } from "./util";

describe("parseInput", () => {
  it("should parse input correctly", () => {
    const input = `this is a user command`;

    const parseResult: InlineEditParseResult = {
      mentions: [],
      mentionQuery: undefined,
    };
    expect(parseUserCommand(input)).to.deep.equal(parseResult);
  });

  it("should parse input correctly", () => {
    const input = `@file1`;

    const parseResult: InlineEditParseResult = {
      mentions: ["file1"],
      mentionQuery: "file1",
    };
    expect(parseUserCommand(input)).to.deep.equal(parseResult);
  });

  it("should parse input correctly", () => {
    const input = `@`;

    const parseResult: InlineEditParseResult = {
      mentions: [],
      mentionQuery: "",
    };
    expect(parseUserCommand(input)).to.deep.equal(parseResult);
  });

  it("should parse input correctly", () => {
    const input = `this is a user command @`;

    const parseResult: InlineEditParseResult = {
      mentions: [],
      mentionQuery: "",
    };
    expect(parseUserCommand(input)).to.deep.equal(parseResult);
  });

  it("should parse input correctly", () => {
    const input = `this is a user command@`;

    const parseResult: InlineEditParseResult = {
      mentions: [],
      mentionQuery: undefined,
    };
    expect(parseUserCommand(input)).to.deep.equal(parseResult);
  });

  it("should parse input correctly", () => {
    const input = ` @file1`;

    const parseResult: InlineEditParseResult = {
      mentions: ["file1"],
      mentionQuery: "file1",
    };
    expect(parseUserCommand(input)).to.deep.equal(parseResult);
  });

  it("should parse input correctly", () => {
    const input = `@file1 `;

    const parseResult: InlineEditParseResult = {
      mentions: ["file1"],
      mentionQuery: undefined,
    };
    expect(parseUserCommand(input)).to.deep.equal(parseResult);
  });

  it("should parse input correctly", () => {
    const input = `@file1 @file2`;
    const parseResult: InlineEditParseResult = {
      mentions: ["file1", "file2"],
      mentionQuery: "file2",
    };
    expect(parseUserCommand(input)).to.deep.equal(parseResult);
  });

  it("should parse input correctly", () => {
    const input = `@file1 @file2 `;
    const parseResult: InlineEditParseResult = {
      mentions: ["file1", "file2"],
      mentionQuery: undefined,
    };
    expect(parseUserCommand(input)).to.deep.equal(parseResult);
  });

  it("should parse input correctly", () => {
    const input = ` @file1 @file2 this is a user command`;
    const parseResult: InlineEditParseResult = {
      mentions: ["file1", "file2"],
      mentionQuery: undefined,
    };
    expect(parseUserCommand(input)).to.deep.equal(parseResult);
  });

  it("should parse input correctly", () => {
    const input = ` @file1 @file2 this is a user command@file3`;
    const parseResult: InlineEditParseResult = {
      mentions: ["file1", "file2"],
      mentionQuery: undefined,
    };
    expect(parseUserCommand(input)).to.deep.equal(parseResult);
  });

  it("should parse input correctly", () => {
    const input = ` @file1 @file2 this is a user command @file3`;
    const parseResult: InlineEditParseResult = {
      mentions: ["file1", "file2", "file3"],
      mentionQuery: "file3",
    };
    expect(parseUserCommand(input)).to.deep.equal(parseResult);
  });

  it("should parse input correctly", () => {
    const input = ` this is a user command @file3 `;
    const parseResult: InlineEditParseResult = {
      mentions: ["file3"],
      mentionQuery: undefined,
    };
    expect(parseUserCommand(input)).to.deep.equal(parseResult);
  });

  it("should parse input correctly", () => {
    const input = ` this is a @file3  user command `;
    const parseResult: InlineEditParseResult = {
      mentions: ["file3"],
      mentionQuery: undefined,
    };
    expect(parseUserCommand(input)).to.deep.equal(parseResult);
  });
});
