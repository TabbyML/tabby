import { documentContext, inline, assertFilterResult } from "./testUtils";
import { dropDuplicated } from "./dropDuplicated";
import { emptyCompletionResultItem } from "../solution";

describe("postprocess", () => {
  describe("dropDuplicated", () => {
    const filter = dropDuplicated();
    it("should drop completion duplicated with suffix", async () => {
      const context = documentContext`javascript
        let sum = (a, b) => {
          ║return a + b;
        };
      `;
      // completion give a `;` at end but context have not
      const completion = inline`
          ├return a + b;┤
      `;
      const expected = emptyCompletionResultItem;
      await assertFilterResult(filter, context, completion, expected);
    });

    it("should drop completion similar to suffix", async () => {
      const context = documentContext`javascript
        let sum = (a, b) => {
          return a + b;
          ║
        };
      `;
      // the difference is a `\n`
      const completion = inline`
          ├}┤
      `;
      const expected = emptyCompletionResultItem;
      await assertFilterResult(filter, context, completion, expected);
    });
  });
});
