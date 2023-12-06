import { expect } from "chai";
import { documentContext, inline } from "./testUtils";
import { dropDuplicated } from "./dropDuplicated";

describe("postprocess", () => {
  describe("dropDuplicated", () => {
    it("should drop completion duplicated with suffix", () => {
      const context = {
        ...documentContext`
        let sum = (a, b) => {
          ║return a + b;
        };
        `,
        language: "javascript",
      };
      // completion give a `;` at end but context have not
      const completion = inline`
          ├return a + b;┤
      `;
      expect(dropDuplicated()(completion, context)).to.be.null;
    });

    it("should drop completion similar to suffix", () => {
      const context = {
        ...documentContext`
        let sum = (a, b) => {
          return a + b;
          ║
        };
        `,
        language: "javascript",
      };
      // the difference is a `\n`
      const completion = inline`
          ├}┤
      `;
      expect(dropDuplicated()(completion, context)).to.be.null;
    });
  });
});
