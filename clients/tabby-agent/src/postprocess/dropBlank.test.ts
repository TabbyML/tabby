import { dropBlank } from "./dropBlank";
import { documentContext, assertFilterResult } from "./testUtils";
import { CompletionItem } from "../CompletionSolution";

describe("postprocess", () => {
  describe("dropBlank", () => {
    const filter = dropBlank();
    const context = documentContext`
      dummy║
    `;
    context.language = "plaintext";

    it("should return null if input is blank", async () => {
      const expected = CompletionItem.createBlankItem(context);
      await assertFilterResult(filter, context, "\n", expected);
      await assertFilterResult(filter, context, "\t\n", expected);
    });
    it("should keep unchanged if input is not blank", async () => {
      const completion = "Not blank";
      const expected = completion;
      await assertFilterResult(filter, context, completion, expected);
    });
  });
});
