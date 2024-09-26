import { documentContext, inline, assertFilterResult } from "./testUtils";
import { dropDuplicated } from "./dropDuplicated";
import { CompletionItem } from "../solution";

describe("postprocess", () => {
  describe("dropDuplicated", () => {
    const filter = dropDuplicated();
    it("should drop completion duplicated with suffix", async () => {
      const context = documentContext`
        let sum = (a, b) => {
          ║return a + b;
        };
      `;
      context.language = "javascript";
      // completion give a `;` at end but context have not
      const completion = inline`
          ├return a + b;┤
      `;
      const expected = CompletionItem.createBlankItem(context);
      await assertFilterResult(filter, context, completion, expected);
    });

    it("should drop completion similar to suffix", async () => {
      const context = documentContext`
        let sum = (a, b) => {
          return a + b;
          ║
        };
      `;
      context.language = "javascript";
      // the difference is a `\n`
      const completion = inline`
          ├}┤
      `;
      const expected = CompletionItem.createBlankItem(context);
      await assertFilterResult(filter, context, completion, expected);
    });
  });
});
