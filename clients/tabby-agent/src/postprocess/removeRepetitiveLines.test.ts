import { expect } from "chai";
import { documentContext, inline } from "./testUtils";
import { removeRepetitiveLines } from "./removeRepetitiveLines";

describe("postprocess", () => {
  describe("removeRepetitiveLines", () => {
    it("should remove repetitive lines", () => {
      const context = {
        ...documentContext`
        function hello() {
          console.log("hello");
        }
        hello();
        hello();
        ║
        `,
        language: "javascript",
      };
      const completion = inline`
        ├hello();
        hello();
        hello();
        hello();
        hello();
        hello();
        hello();
        hello();
        hello();
        hello();┤
        `;
      const expected = inline`
        ├hello();┤
      `;
      expect(removeRepetitiveLines()(completion, context)).to.eq(expected);
    });

    it("should remove repetitive lines with patterns", () => {
      const context = {
        ...documentContext`
        const a = 1;
        ║
        `,
        language: "javascript",
      };
      const completion = inline`
        ├const b = 1;
        const c = 1;
        const d = 1;
        const e = 1;
        const f = 1;
        const g = 1;
        const h = 1;
        const i = 1;
        const j = 1;
        const k =┤`;
      const expected = inline`
        ├const b = 1;┤
        `;
      expect(removeRepetitiveLines()(completion, context)).to.eq(expected);
    });
  });
});
