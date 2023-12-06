import { expect } from "chai";
import { documentContext, inline } from "./testUtils";
import { trimSpace } from "./trimSpace";

describe("postprocess", () => {
  describe("trimSpace", () => {
    it("should remove trailing space", () => {
      const context = {
        ...documentContext`
        let foo = new ║
        `,
        language: "javascript",
      };
      const completion = inline`
                      ├Foo(); ┤
        `;
      const expected = inline`
                      ├Foo();┤
      `;
      expect(trimSpace()(completion, context)).to.eq(expected);
    });

    it("should not remove trailing space if filling in line", () => {
      const context = {
        ...documentContext`
        let foo = sum(║baz)
        `,
        language: "javascript",
      };
      const completion = inline`
                      ├bar, ┤
        `;
      expect(trimSpace()(completion, context)).to.eq(completion);
    });

    it("should remove trailing space if filling in line with suffix starts with space", () => {
      const context = {
        ...documentContext`
        let foo = sum(║ baz)
        `,
        language: "javascript",
      };
      const completion = inline`
                      ├bar, ┤
        `;
      const expected = inline`
                      ├bar,┤
        `;
      expect(trimSpace()(completion, context)).to.eq(expected);
    });

    it("should not remove leading space if current line is blank", () => {
      const context = {
        ...documentContext`
        function sum(a, b) {
        ║
        }
        `,
        language: "javascript",
      };
      const completion = inline`
        ├  return a + b;┤
        `;
      expect(trimSpace()(completion, context)).to.eq(completion);
    });

    it("should remove leading space if current line is not blank and ends with space", () => {
      const context = {
        ...documentContext`
        let foo = ║
        `,
        language: "javascript",
      };
      const completion = inline`
                  ├ sum(bar, baz);┤
        `;
      const expected = inline`
                  ├sum(bar, baz);┤
        `;
      expect(trimSpace()(completion, context)).to.eq(expected);
    });
  });
});
