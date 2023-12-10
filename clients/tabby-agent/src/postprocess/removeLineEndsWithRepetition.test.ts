import { expect } from "chai";
import { documentContext, inline } from "./testUtils";
import { removeLineEndsWithRepetition } from "./removeLineEndsWithRepetition";

describe("postprocess", () => {
  describe("removeLineEndsWithRepetition", () => {
    it("should drop one line completion ends with repetition", () => {
      const context = {
        ...documentContext`
        let foo = ║
        `,
        language: "javascript",
      };
      const completion = inline`
                  ├foo = foo = foo = foo = foo = foo = foo =┤
        `;
      expect(removeLineEndsWithRepetition()(completion, context)).to.be.null;
    });

    it("should remove last line that ends with repetition", () => {
      const context = {
        ...documentContext`
        let largeNumber = 1000000
        let veryLargeNumber = ║
        `,
        language: "javascript",
      };
      const completion = inline`
                              ├1000000000
        let superLargeNumber = 1000000000000000000000000000000000000000000000┤
        `;
      const expected = inline`
                              ├1000000000┤
        `;
      expect(removeLineEndsWithRepetition()(completion, context)).to.eq(expected);
    });

    it("should keep repetition less than threshold", () => {
      const context = {
        ...documentContext`
        let largeNumber = 1000000
        let veryLargeNumber = ║
        `,
        language: "javascript",
      };
      const completion = inline`
                              ├1000000000000┤
        `;
      const expected = completion;
      expect(removeLineEndsWithRepetition()(completion, context)).to.eq(expected);
    });
  });
});
