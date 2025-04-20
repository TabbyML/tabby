import { documentContext, inline, assertFilterResult } from "./testUtils";
import { trimSpace } from "./trimSpace";

describe("postprocess", () => {
  describe("trimSpace", () => {
    const filter = trimSpace();
    it("should remove trailing space", async () => {
      const context = documentContext`javascript
        let foo = new ║
      `;
      const completion = inline`
                      ├Foo(); ┤
        `;
      const expected = inline`
                      ├Foo();┤
      `;
      await assertFilterResult(filter, context, completion, expected);
    });

    it("should not remove trailing space if filling in line", async () => {
      const context = documentContext`javascript
        let foo = sum(║baz)
      `;
      const completion = inline`
                      ├bar, ┤
      `;
      const expected = completion;
      await assertFilterResult(filter, context, completion, expected);
    });

    it("should remove trailing space if filling in line with suffix starts with space", async () => {
      const context = documentContext`javascript
        let foo = sum(║ baz)
      `;
      const completion = inline`
                      ├bar, ┤
      `;
      const expected = inline`
                      ├bar,┤
      `;
      await assertFilterResult(filter, context, completion, expected);
    });

    it("should not remove leading space if current line is blank", async () => {
      const context = documentContext`javascript
        function sum(a, b) {
        ║
        }
      `;
      const completion = inline`
        ├  return a + b;┤
      `;
      const expected = completion;
      await assertFilterResult(filter, context, completion, expected);
    });

    it("should remove leading space if current line is not blank and ends with space", async () => {
      const context = documentContext`javascript
        let foo = ║
      `;
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
