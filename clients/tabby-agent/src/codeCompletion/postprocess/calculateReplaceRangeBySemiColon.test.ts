import { documentContext, inline, assertFilterResult } from "./testUtils";
import { calculateReplaceRangeBySemiColon } from "./calculateReplaceRangeBySemiColon";

describe("postprocess", () => {
  describe("calculateReplaceRangeBySemiColon", () => {
    const filter = calculateReplaceRangeBySemiColon;

    it("should handle semicolon in string concatenation", async () => {
      const context = documentContext`
          const content = "hello world";
          const a = "nihao" + ║;
        `;
      context.language = "typescript";
      const completion = {
        text: inline`
                                 ├content;┤
          `,
      };
      const expected = {
        text: inline`
                                 ├content;┤
          `,
        replaceSuffix: ";",
      };
      await assertFilterResult(filter, context, completion, expected);
    });

    it("should handle semicolon at the end of a statement", async () => {
      const context = documentContext`
          const content = "hello world"║;
        `;
      context.language = "typescript";
      const completion = {
        text: inline`
                                 ├;┤
          `,
      };
      const expected = {
        text: inline`
                                 ├;┤
          `,
        replaceSuffix: ";",
      };
      await assertFilterResult(filter, context, completion, expected);
    });

    it("should not handle any semicolon at the end of a statement", async () => {
      const context = documentContext`
          const content = "hello world"║
        `;
      context.language = "typescript";
      const completion = {
        text: inline`
                                 ├┤
          `,
      };
      const expected = {
        text: inline`
                                 ├┤
          `,
        replaceSuffix: "",
      };
      await assertFilterResult(filter, context, completion, expected);
    });

    it("should not modify if no semicolon in completion text", async () => {
      const context = documentContext`
          const content = "hello world"║
        `;
      context.language = "typescript";
      const completion = {
        text: inline`
                                 ├content┤
          `,
      };
      const expected = {
        text: inline`
                                 ├content┤
          `,
        replaceSuffix: "",
      };
      await assertFilterResult(filter, context, completion, expected);
    });

    it("should handle multiple semicolons in completion text", async () => {
      const context = documentContext`
          const content = "hello world"║;
        `;
      context.language = "typescript";
      const completion = {
        text: inline`
                                 ├content;;┤
          `,
      };
      const expected = {
        text: inline`
                                 ├content;;┤
          `,
        replaceSuffix: ";",
      };
      await assertFilterResult(filter, context, completion, expected);
    });

    it("should handle semicolon in the middle of a statement", async () => {
      const context = documentContext`
          const content = "hello; world"║;
        `;
      context.language = "typescript";
      const completion = {
        text: inline`
                                 ├content;┤
          `,
      };
      const expected = {
        text: inline`
                                 ├content;┤
          `,
        replaceSuffix: ";",
      };
      await assertFilterResult(filter, context, completion, expected);
    });
  });
});
