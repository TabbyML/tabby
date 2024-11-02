import { documentContext, inline, assertFilterResult } from "./testUtils";
import { normalizeIndentation } from "./normalizeIndentation";

describe("postprocess", () => {
  describe("normalizeIndentation", () => {
    const filter = normalizeIndentation();

    it("should normalize indentation when cursor is on empty line", async () => {
      const context = documentContext`
        function test() {
          ║
        }
      `;
      const completion = inline`
          ├console.log("test");┤
      `;
      const expected = inline`
          ├console.log("test");┤
      `;
      await assertFilterResult(filter, context, completion, expected);
    });

    it("should not modify indentation when cursor line has content", async () => {
      const context = documentContext`
        let x = ║value
      `;
      const completion = inline`
                ├42┤
      `;
      const expected = inline`
                ├42┤
      `;
      await assertFilterResult(filter, context, completion, expected);
    });

    it("should handle empty lines in nested blocks", async () => {
      const context = documentContext`
        if (true) {
          while (condition) {
            ║
          }
        }
      `;
      context.language = "javascript";
      const completion = inline`
            ├doSomething();┤
      `;
      const expected = inline`
            ├doSomething();┤
      `;
      await assertFilterResult(filter, context, completion, expected);
    });

    it("should handle deep nested block indentation", async () => {
      const context = documentContext`
        if let Some(code_query) = code_query {
            let hits = self.collect_relevant_code(
                &context_info_helper,
                code_query,
            ).await?;
            ║
        }
      `;
      context.language = "rust";
      const completion = inline`
            ├attachment.code_hits = Some(hits);┤
      `;
      const expected = inline`
            ├attachment.code_hits = Some(hits);┤
      `;
      await assertFilterResult(filter, context, completion, expected);
    });

    it("should handle method arguments with object literals", async () => {
      const context = documentContext`
        await client.request(
        ║
        );
      `;
      context.language = "javascript";
      const completion = inline`
        ├{
            method: 'POST',
            url: '/api/data',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
          }┤
      `;
      const expected = inline`
        ├{
            method: 'POST',
            url: '/api/data',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
          }┤
      `;
      await assertFilterResult(filter, context, completion, expected);
    });
  });
});
