import { expect } from "chai";
import { documentContext, inline } from "./testUtils";
import { removeOverlapping } from "./removeOverlapping";

describe("postprocess", () => {
  describe("removeOverlapping", () => {
    it("should remove content overlapped between completion and suffix", () => {
      const context = {
        ...documentContext`
        function sum(a, b) {
          ║
          return value;
        }
        `,
        language: "javascript",
      };
      const completion = inline`
          ├let value = a + b;
          return value;
        }┤
        `;
      const expected = inline`
          ├let value = a + b;┤
        ┴┴
        `;
      expect(removeOverlapping(context)(completion)).to.eq(expected);
    });

    // Bad case
    it("can not remove text that suffix not exactly starts with", () => {
      const context = {
        ...documentContext`
        let sum = (a, b) => {
          ║return a + b;
        }
        `,
        language: "javascript",
      };
      // completion give a `;` at end but context have not
      const completion = inline`
          ├return a + b;
        };┤
      `;
      expect(removeOverlapping(context)(completion)).to.eq(completion);
    });

    // Bad case
    it("can not remove text that suffix not exactly starts with", () => {
      const context = {
        ...documentContext`
        let sum = (a, b) => {
          return a + b;
          ║
        }
        `,
        language: "javascript",
      };
      // the difference is a `\n`
      const completion = inline`
          ├}┤
      `;
      expect(removeOverlapping(context)(completion)).to.eq(completion);
    });
  });
});
