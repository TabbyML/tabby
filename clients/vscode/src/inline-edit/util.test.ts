import { describe } from "mocha";
import { expect } from "chai";
import { parseUserCommand, InlineEditParseResult, MentionType } from "./util";

describe("parseInput", () => {
  it("should parse input correctly", () => {
    const input = `this is a user command`;

    const parseResult: InlineEditParseResult = {
      mentions: [],
      mentionQuery: undefined,
      mentionType: undefined,
    };
    expect(parseUserCommand(input)).to.deep.equal(parseResult);
  });

  it("should parse input correctly", () => {
    const input = `@file1`;

    const parseResult: InlineEditParseResult = {
      mentions: [{ text: "file1", type: MentionType.File }],
      mentionQuery: "file1",
      mentionType: MentionType.File,
    };
    expect(parseUserCommand(input)).to.deep.equal(parseResult);
  });

  it("should parse input correctly", () => {
    const input = `@`;

    const parseResult: InlineEditParseResult = {
      mentions: [],
      mentionQuery: "",
      mentionType: MentionType.File,
    };
    expect(parseUserCommand(input)).to.deep.equal(parseResult);
  });

  it("should parse input correctly", () => {
    const input = `this is a user command @`;

    const parseResult: InlineEditParseResult = {
      mentions: [],
      mentionQuery: "",
      mentionType: MentionType.File,
    };
    expect(parseUserCommand(input)).to.deep.equal(parseResult);
  });

  it("should parse input correctly", () => {
    const input = `this is a user command@`;

    const parseResult: InlineEditParseResult = {
      mentions: [],
      mentionQuery: undefined,
      mentionType: undefined,
    };
    expect(parseUserCommand(input)).to.deep.equal(parseResult);
  });

  it("should parse input correctly", () => {
    const input = ` @file1`;

    const parseResult: InlineEditParseResult = {
      mentions: [{ text: "file1", type: MentionType.File }],
      mentionQuery: "file1",
      mentionType: MentionType.File,
    };
    expect(parseUserCommand(input)).to.deep.equal(parseResult);
  });

  it("should parse input correctly", () => {
    const input = `@file1 `;

    const parseResult: InlineEditParseResult = {
      mentions: [{ text: "file1", type: MentionType.File }],
      mentionQuery: undefined,
      mentionType: undefined,
    };
    expect(parseUserCommand(input)).to.deep.equal(parseResult);
  });

  it("should parse input correctly", () => {
    const input = `@file1 @file2`;
    const parseResult: InlineEditParseResult = {
      mentions: [
        { text: "file1", type: MentionType.File },
        { text: "file2", type: MentionType.File },
      ],
      mentionQuery: "file2",
      mentionType: MentionType.File,
    };
    expect(parseUserCommand(input)).to.deep.equal(parseResult);
  });

  it("should parse input correctly", () => {
    const input = `@file1 @file2 `;
    const parseResult: InlineEditParseResult = {
      mentions: [
        { text: "file1", type: MentionType.File },
        { text: "file2", type: MentionType.File },
      ],
      mentionQuery: undefined,
      mentionType: undefined,
    };
    expect(parseUserCommand(input)).to.deep.equal(parseResult);
  });

  it("should parse input correctly", () => {
    const input = ` @file1 @file2 this is a user command`;
    const parseResult: InlineEditParseResult = {
      mentions: [
        { text: "file1", type: MentionType.File },
        { text: "file2", type: MentionType.File },
      ],
      mentionQuery: undefined,
      mentionType: undefined,
    };
    expect(parseUserCommand(input)).to.deep.equal(parseResult);
  });

  it("should parse input correctly", () => {
    const input = ` @file1 @file2 this is a user command@file3`;
    const parseResult: InlineEditParseResult = {
      mentions: [
        { text: "file1", type: MentionType.File },
        { text: "file2", type: MentionType.File },
      ],
      mentionQuery: undefined,
      mentionType: undefined,
    };
    expect(parseUserCommand(input)).to.deep.equal(parseResult);
  });

  it("should parse input correctly", () => {
    const input = ` @file1 @file2 this is a user command @file3`;
    const parseResult: InlineEditParseResult = {
      mentions: [
        { text: "file1", type: MentionType.File },
        { text: "file2", type: MentionType.File },
        { text: "file3", type: MentionType.File },
      ],
      mentionQuery: "file3",
      mentionType: MentionType.File,
    };
    expect(parseUserCommand(input)).to.deep.equal(parseResult);
  });

  it("should parse input correctly", () => {
    const input = ` this is a user command @file3 `;
    const parseResult: InlineEditParseResult = {
      mentions: [{ text: "file3", type: MentionType.File }],
      mentionQuery: undefined,
      mentionType: undefined,
    };
    expect(parseUserCommand(input)).to.deep.equal(parseResult);
  });

  it("should parse input correctly", () => {
    const input = ` this is a @file3  user command `;
    const parseResult: InlineEditParseResult = {
      mentions: [{ text: "file3", type: MentionType.File }],
      mentionQuery: undefined,
      mentionType: undefined,
    };
    expect(parseUserCommand(input)).to.deep.equal(parseResult);
  });
});
