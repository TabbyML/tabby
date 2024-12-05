import { dropMinimum } from "./dropMinimum";
import { documentContext, assertFilterResult } from "./testUtils";
import { CompletionItem } from "../solution";

describe("postprocess", () => {
  describe("dropMinimum", () => {
    const filter = dropMinimum({ limitScope: null, minCompletionChars: 4, calculateReplaceRange: null });
    const context = documentContext`
      dummy║
    `;
    context.language = "plaintext";

    it("should return null if input is < 4 non-whitespace characters", async () => {
      const expected = CompletionItem.createBlankItem(context);
      await assertFilterResult(filter, context, "\n", expected);
      await assertFilterResult(filter, context, "\t\n", expected);
      await assertFilterResult(filter, context, "ab\t\n", expected);
    });
    it("should keep unchanged if input is >= 4 non-whitespace characters", async () => {
      const completion = "Greater than 4";
      const expected = completion;
      await assertFilterResult(filter, context, completion, expected);
    });
  });
});
