import { documentContext, inline, assertFilterResult } from "./testUtils";
import { removeRepetitiveBlocks } from "./removeRepetitiveBlocks";

describe("postprocess", () => {
  describe("removeRepetitiveBlocks", () => {
    const filter = removeRepetitiveBlocks();
    it("should remove repetitive blocks", async () => {
      const context = documentContext`
        function myFuncA() {
          console.log("myFuncA called.");
        }

        ║
      `;
      context.language = "javascript";
      const completion = inline`
        ├function myFuncB() {
          console.log("myFuncB called.");
        }

        function myFuncC() {
          console.log("myFuncC called.");
        }

        function myFuncD() {
          console.log("myFuncD called.");
        }

        function myFuncE() {
          console.log("myFuncE called.");
        }

        function myFuncF() {
          console.log("myFuncF called.");
        }

        function myFuncG() {
          console.log("myFuncG called.");
        }

        function myFuncH() {
          console.log("myFuncH ┤
      `;
      const expected = inline`
        ├function myFuncB() {
          console.log("myFuncB called.");
        }┤
      `;
      await assertFilterResult(filter, context, completion, expected);
    });
  });
});
