import { documentContext, inline, assertFilterResult } from "./testUtils";
import { trimSpace } from "./trimSpace";

describe("postprocess", () => {
  describe("trimSpace", () => {
    const filter = trimSpace();
    it("should remove trailing space", async () => {
      const context = documentContext`
        let foo = new ║
      `;
      context.language = "javascript";
      const completion = inline`
                      ├Foo(); ┤
        `;
      const expected = inline`
                      ├Foo();┤
      `;
      await assertFilterResult(filter, context, completion, expected);
    });

    it("should not remove trailing space if filling in line", async () => {
      const context = documentContext`
        let foo = sum(║baz)
      `;
      context.language = "javascript";
      const completion = inline`
                      ├bar, ┤
      `;
      const expected = completion;
      await assertFilterResult(filter, context, completion, expected);
    });

    it("should remove trailing space if filling in line with suffix starts with space", async () => {
      const context = documentContext`
        let foo = sum(║ baz)
      `;
      context.language = "javascript";
      const completion = inline`
                      ├bar, ┤
      `;
      const expected = inline`
                      ├bar,┤
      `;
      await assertFilterResult(filter, context, completion, expected);
    });

    it("should not remove leading space if current line is blank", async () => {
      const context = documentContext`
        function sum(a, b) {
        ║
        }
      `;
      context.language = "javascript";
      const completion = inline`
        ├  return a + b;┤
      `;
      const expected = completion;
      await assertFilterResult(filter, context, completion, expected);
    });

    it("should remove leading space if current line is not blank and ends with space", async () => {
      const context = documentContext`
        let foo = ║
      `;
      context.language = "javascript";
      const completion = inline`
                  ├ sum(bar, baz);┤
      `;
      const expected = inline`
                  ├sum(bar, baz);┤
      `;
      await assertFilterResult(filter, context, completion, expected);
    });
  });
});
