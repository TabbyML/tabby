import { documentContext, inline, assertFilterResult } from "./testUtils";
import { normalizeIndentation } from "./normalizeIndentation";

describe("postprocess", () => {
  describe("normalizeIndentation", () => {
    const filter = normalizeIndentation();

    it("should fix first line extra indentation", async () => {
      const context = documentContext`
        function test() {
          ║
        }
      `;
      const completion = inline`
           ├ const x = 1;
          const y = 2;┤
      `;
      const expected = inline`
          ├const x = 1;
          const y = 2;┤
      `;
      await assertFilterResult(filter, context, completion, expected);
    });
    it("should remove extra indent", async () => {
      const context = documentContext`
        if (true) {
          if (condition) {
            ║
          }
        }
      `;
      const completion = inline`
            ├doSomething();
             doAnother();┤
      `;
      const expected = inline`
            ├doSomething();
            doAnother();┤
      `;
      await assertFilterResult(filter, context, completion, expected);
    });

    it("should handle both inappropriate first line and extra indent case 01", async () => {
      const context = documentContext`
        if (true) {
          if (condition) {
            ║
          }
        }
      `;
      const completion = inline`
            ├ doSomething();
             doAnother();┤
      `;
      const expected = inline`
           ├doSomething();
           doAnother();┤
      `;
      await assertFilterResult(filter, context, completion, expected);
    });

    it("should handle both inappropriate extra indent case 02", async () => {
      const context = documentContext`
        {
          "command": "test",
          ║
        }
      `;
      const completion = inline`
          ├"title": "Run Test",
           "category": "Tabby"┤
      `;
      const expected = inline`
          ├"title": "Run Test",
          "category": "Tabby"┤
      `;
      await assertFilterResult(filter, context, completion, expected);
    });

    it("should do nothing", async () => {
      const context = documentContext`
        function foo() {
        ║
        }
      `;
      const completion = inline`
        ├    bar();┤
      `;
      const expected = inline`
        ├    bar();┤
      `;
      await assertFilterResult(filter, context, completion, expected);
    });
    it("should fix extra indent", async () => {
      const context = documentContext`
        {
         ║
        }
      `;
      const completion = inline`
        ├"command": "tabby.inlineCompletion.trigger.automatic",
         "title":  "Trigger Code Completion Automatically",
         "category": "Tabby"┤
      `;
      const expected = inline`
        ├"command": "tabby.inlineCompletion.trigger.automatic",
        "title":  "Trigger Code Completion Automatically",
        "category": "Tabby"┤
      `;
      await assertFilterResult(filter, context, completion, expected);
    });

    it("shouldn't change indent", async () => {
      const context = documentContext`
        {
          "command": "tabby.toggleInlineCompletionTriggerMode",
          "title": "Toggle Code Completion Trigger Mode (Automatic/Manual)",
          "category": "Tabby"
        },
        ║
      `;
      const completion = inline`
        ├{
          "command": "tabby.inlineCompletion.trigger",
          "title": "Trigger Code Completion Manually",
          "category": "Tabby"
        },┤
      `;
      const expected = inline`
        ├{
          "command": "tabby.inlineCompletion.trigger",
          "title": "Trigger Code Completion Manually",
          "category": "Tabby"
        },┤
      `;
      await assertFilterResult(filter, context, completion, expected);
    });
  });
});
