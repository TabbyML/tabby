import { expect } from "chai";
import { documentContext, inline } from "./testUtils";
import { removeDuplicatedBlockClosingLine } from "./removeDuplicatedBlockClosingLine";

describe("postprocess", () => {
  describe("removeDuplicatedBlockClosingLine", () => {
    it("should remove duplicated block closing line.", () => {
      const context = {
        ...documentContext`
        function hello() {
          ║
        }
        `,
        language: "javascript",
      };
      const completion = inline`
          ├console.log("hello");
        }┤
      `;
      const expected = inline`
          ├console.log("hello");┤
        ┴┴
      `;
      expect(removeDuplicatedBlockClosingLine()(completion, context)).to.eq(expected);
    });

    it("should remove duplicated block closing line.", () => {
      const context = {
        ...documentContext`
        function check(condition) {
          if (!condition) {
            ║
          } else {
            return;
          }
        }
        `,
        language: "javascript",
      };
      const completion = inline`
            ├throw new Error("check not passed");
          }┤
        ┴┴
      `;
      const expected = inline`
            ├throw new Error("check not passed");┤
        ┴┴┴┴
      `;
      expect(removeDuplicatedBlockClosingLine()(completion, context)).to.eq(expected);
    });

    it("should not remove non-duplicated block closing line.", () => {
      const context = {
        ...documentContext`
        function check(condition) {
          if (!condition) {
            ║
        }
        `,
        language: "javascript",
      };
      const completion = inline`
            ├throw new Error("check not passed");
          }┤
        ┴┴
      `;
      expect(removeDuplicatedBlockClosingLine()(completion, context)).to.eq(completion);
    });
  });
});
