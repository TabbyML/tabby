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
      expect(dropDuplicated(context)(completion)).to.be.null;
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
      expect(dropDuplicated(context)(completion)).to.be.null;
    });

    it("should drop completion that first 3 lines are similar to suffix", () => {
      const context = {
        ...documentContext`
        var a, b;
        // swap a and b║
        let z = a;
        a = b;
        b = z;
        // something else
        `,
        language: "javascript",
      };
      const completion = inline`
                       ├
        let c = a;
        a = b;
        b = c;
        console.log({a, b});┤
      `;
      expect(dropDuplicated(context)(completion)).to.be.null;
    });
  });
});
