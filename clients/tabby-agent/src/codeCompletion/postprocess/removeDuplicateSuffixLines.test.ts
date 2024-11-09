import { documentContext, inline, assertFilterResult } from "./testUtils";
import { removeDuplicateSuffixLines } from "./removeDuplicateSuffixLines";

describe("postprocess", () => {
  describe("removeDuplicateSuffixLines", () => {
    const filter = removeDuplicateSuffixLines();

    it("should remove duplicated suffix lines", async () => {
      const context = documentContext`
        function example() {
          const items = [
            ║
          ];
        }
      `;
      context.language = "javascript";
      const completion = inline`
            ├1,
            2,
            3,
            4,┤
      `;
      context.suffix = `
            4,
            5,
            6
          ];
        }
      `;
      const expected = inline`
            ├1,
            2,
            3,┤
      `;
      await assertFilterResult(filter, context, completion, expected);
    });

    it("should handle empty suffix", async () => {
      const context = documentContext`
        const value = ║
      `;
      context.language = "javascript";
      const completion = inline`
                      ├42;┤
      `;
      context.suffix = "";
      const expected = completion;
      await assertFilterResult(filter, context, completion, expected);
    });

    it("should handle multiple line matches", async () => {
      const context = documentContext`
        class Example {
          constructor() {
            ║
          }
        }
      `;
      context.language = "javascript";
      const completion = inline`
            ├this.value = 1;
            this.name = "test";
            this.items = [];
            this.setup();┤
      `;
      context.suffix = `
            this.setup();
            this.init();
          }
        }
      `;
      const expected = inline`
            ├this.value = 1;
            this.name = "test";
            this.items = [];┤
      `;
      await assertFilterResult(filter, context, completion, expected);
    });

    it("should handle partial line matches without trimming", async () => {
      const context = documentContext`
        const config = {
          ║
        };
      `;
      context.language = "javascript";
      const completion = inline`
          ├name: "test",
          value: 42,
          items: [],
          enabled: true,┤
      `;
      context.suffix = `
          enabled: true,
          debug: false
        };
      `;
      const expected = inline`
          ├name: "test",
          value: 42,
          items: [],┤
      `;
      await assertFilterResult(filter, context, completion, expected);
    });

    it("should not modify when no matches found", async () => {
      const context = documentContext`
        function process() {
          ║
        }
      `;
      context.language = "javascript";
      const completion = inline`
          ├console.log("processing");
          return true;┤
      `;
      context.suffix = `
          console.log("done");
        }
      `;
      const expected = completion;
      await assertFilterResult(filter, context, completion, expected);
    });

    it("should handle whitespace differences", async () => {
      const context = documentContext`
        const arr = [
          ║
        ];
      `;
      context.language = "javascript";
      const completion = inline`
          ├1,
             2,
               3,
                 4,┤
      `;
      context.suffix = `
                 4,
                 5,
                 6
        ];
      `;
      const expected = inline`
          ├1,
             2,
               3,┤
      `;
      await assertFilterResult(filter, context, completion, expected);
    });
  });
});
