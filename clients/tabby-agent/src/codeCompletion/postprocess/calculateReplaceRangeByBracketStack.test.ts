import { documentContext, inline, assertFilterResult, assertFilterResultNotEqual } from "./testUtils";
import { calculateReplaceRangeByBracketStack } from "./calculateReplaceRangeByBracketStack";

describe("postprocess", () => {
  describe("calculateReplaceRangeByBracketStack", () => {
    const filter = calculateReplaceRangeByBracketStack;
    it("should handle auto closing quotes", async () => {
      const context = documentContext`
        const hello = "║"
      `;
      context.language = "typescript";
      const completion = {
        text: inline`
                       ├hello";┤
        `,
      };
      const expected = {
        text: inline`
                       ├hello";┤
        `,
        replaceSuffix: '"',
      };
      await assertFilterResult(filter, context, completion, expected);
    });

    it("should handle auto closing quotes", async () => {
      const context = documentContext`
        let htmlMarkup = \`║\`
      `;
      context.language = "typescript";
      const completion = {
        text: inline`
                           ├<h1>\${message}</h1>\`;┤
        `,
      };
      const expected = {
        text: inline`
                           ├<h1>\${message}</h1>\`;┤
        `,
        replaceSuffix: "`",
      };
      await assertFilterResult(filter, context, completion, expected);
    });

    it("should handle multiple auto closing brackets", async () => {
      const context = documentContext`
        process.on('data', (data) => {║})
      `;
      context.language = "typescript";
      const completion = {
        text: inline`
                                      ├
          console.log(data);
        });┤
        `,
      };
      const expected = {
        text: inline`
                                      ├
          console.log(data);
        });┤
        `,
        replaceSuffix: "})",
      };
      await assertFilterResult(filter, context, completion, expected);
    });

    it("should handle multiple auto closing brackets", async () => {
      const context = documentContext`
        let mat: number[][][] = [[[║]]]
      `;
      context.language = "typescript";
      const completion = {
        text: inline`
                                   ├1, 2], [3, 4]], [[5, 6], [7, 8]]];┤
        `,
      };
      const expected = {
        text: inline`
                                   ├1, 2], [3, 4]], [[5, 6], [7, 8]]];┤
        `,
        replaceSuffix: "]]]",
      };
      await assertFilterResult(filter, context, completion, expected);
    });

    it("should handle html tags", async () => {
      const context = documentContext`
        <html></h║>
      `;
      context.language = "html";
      const completion = {
        text: inline`
                 ├tml>┤
        `,
      };
      const expected = {
        text: inline`
                 ├tml>┤
        `,
        replaceSuffix: ">",
      };
      await assertFilterResult(filter, context, completion, expected);
    });

    it("should handle jsx tags", async () => {
      const context = documentContext`
        root.render(
          <React.StrictMode>
            <App m║/>
          </React.StrictMode>
        );
      `;
      context.language = "javascriptreact";
      const completion = {
        text: inline`
                  ├essage={message} />┤
        `,
      };
      const expected = {
        text: inline`
                  ├essage={message} />┤
        `,
        replaceSuffix: "/>",
      };
      await assertFilterResult(filter, context, completion, expected);
    });
  });

  describe("calculateReplaceRangeByBracketStack: bad cases", () => {
    const filter = calculateReplaceRangeByBracketStack;
    it("cannot handle the case of completion bracket stack is same with suffix but should not be replaced", async () => {
      const context = documentContext`
        function clamp(n: number, max: number, min: number): number {
          return Math.max(Math.min(║);
        }
      `;
      context.language = "typescript";
      const completion = {
        text: inline`
                                   ├n, max), min┤
        `,
      };
      const expected = completion;
      await assertFilterResultNotEqual(filter, context, completion, expected);
    });
  });
});
