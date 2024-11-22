import { documentContext, inline, assertFilterResult } from "./testUtils";
import { removeDuplicateSuffixLines } from "./removeDuplicateSuffixLines";

describe("removeDuplicateSuffixLines", () => {
  const filter = removeDuplicateSuffixLines();

  it("should remove duplicated array item", async () => {
    const context = documentContext`
      const items = [
        ║
        { id: 2 },
        { id: 3 }
      ];
    `;
    const completion = inline`
      ├{ id: 1 },
      { id: 2 },┤
    `;
    const expected = inline`
      ├{ id: 1 },┤
    `;
    await assertFilterResult(filter, context, completion, expected);
  });

  it("should handle empty content after cursor", async () => {
    const context = documentContext`
      const a = ║
    `;
    const completion = inline`
      ├42;┤
    `;
    const expected = completion;
    await assertFilterResult(filter, context, completion, expected);
  });

  it("should remove duplicated comma items", async () => {
    const context = documentContext`
      function example() {
        const items = [
          ║
          4,
          5,
          6
        ];
      }
    `;
    const completion = inline`
      ├1,
      2,
      3,
      4,┤
    `;
    const expected = inline`
      ├1,
      2,
      3,┤
    `;
    await assertFilterResult(filter, context, completion, expected);
  });

  it("should remove duplicate method calls", async () => {
    const context = documentContext`
      class Example {
        constructor() {
          ║
          this.setup();
          this.init()
        }
      }
    `;
    const completion = inline`
      ├this.value = 1;
      this.name = "test";
      this.items = [];
      this.setup();┤
    `;
    const expected = inline`
      ├this.value = 1;
      this.name = "test";
      this.items = [];┤
    `;
    await assertFilterResult(filter, context, completion, expected);
  });

  it("should remove duplicate object properties", async () => {
    const context = documentContext`
      const config = {
        ║
        enabled: true,
        debug: false          
      };
    `;
    const completion = inline`
      ├name: "test",
      value: 42,
      items: [],
      enabled: true,┤
    `;
    const expected = inline`
      ├name: "test",
      value: 42,
      items: [],┤
    `;
    await assertFilterResult(filter, context, completion, expected);
  });

  it("should keep content when no matches", async () => {
    const context = documentContext`
      function process() {
        ║
        console.log("done");          
      }
    `;
    const completion = inline`
      ├console.log("processing");┤
    `;
    const expected = completion;
    await assertFilterResult(filter, context, completion, expected);
  });
});
