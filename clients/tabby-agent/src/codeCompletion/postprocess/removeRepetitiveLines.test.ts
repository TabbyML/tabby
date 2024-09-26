import { documentContext, inline, assertFilterResult } from "./testUtils";
import { removeRepetitiveLines } from "./removeRepetitiveLines";

describe("postprocess", () => {
  describe("removeRepetitiveLines", () => {
    const filter = removeRepetitiveLines();
    it("should remove repetitive lines", async () => {
      const context = documentContext`
        function hello() {
          console.log("hello");
        }
        hello();
        hello();
        ║
      `;
      context.language = "javascript";
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
      await assertFilterResult(filter, context, completion, expected);
    });

    it("should remove repetitive lines with patterns", async () => {
      const context = documentContext`
        const a = 1;
        ║
      `;
      context.language = "javascript";
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
      await assertFilterResult(filter, context, completion, expected);
    });
  });
});
