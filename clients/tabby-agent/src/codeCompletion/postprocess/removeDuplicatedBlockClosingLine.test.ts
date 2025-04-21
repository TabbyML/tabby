import { documentContext, inline, assertFilterResult } from "./testUtils";
import { removeDuplicatedBlockClosingLine } from "./removeDuplicatedBlockClosingLine";

describe("postprocess", () => {
  describe("removeDuplicatedBlockClosingLine", () => {
    const filter = removeDuplicatedBlockClosingLine();
    it("should remove duplicated block closing line.", async () => {
      const context = documentContext`javascript
        function hello() {
          ║
        }
      `;
      const completion = inline`
          ├console.log("hello");
        }┤
      `;
      const expected = inline`
          ├console.log("hello");┤
        ┴┴
      `;
      await assertFilterResult(filter, context, completion, expected);
    });

    it("should remove duplicated block closing line.", async () => {
      const context = documentContext`javascript
        function check(condition) {
          if (!condition) {
            ║
          } else {
            return;
          }
        }
      `;
      const completion = inline`
            ├throw new Error("check not passed");
          }┤
        ┴┴
      `;
      const expected = inline`
            ├throw new Error("check not passed");┤
        ┴┴┴┴
      `;
      await assertFilterResult(filter, context, completion, expected);
    });

    it("should not remove non-duplicated block closing line.", async () => {
      const context = documentContext`javascript
        function check(condition) {
          if (!condition) {
            ║
        }
      `;
      const completion = inline`
            ├throw new Error("check not passed");
          }┤
        ┴┴
      `;
      const expected = completion;
      await assertFilterResult(filter, context, completion, expected);
    });
  });
});
