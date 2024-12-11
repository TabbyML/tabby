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
  });
});
