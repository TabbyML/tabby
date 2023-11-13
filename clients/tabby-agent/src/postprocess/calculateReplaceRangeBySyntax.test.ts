import { expect } from "chai";
import { documentContext, inline } from "./testUtils";
import { calculateReplaceRangeBySyntax } from "./calculateReplaceRangeBySyntax";

describe("postprocess", () => {
  describe("calculateReplaceRangeBySyntax", () => {
    it("should handle auto closing quotes", async () => {
      const context = {
        ...documentContext`
        const hello = "║"
        `,
        language: "typescript",
      };
      const response = {
        id: "",
        choices: [
          {
            index: 0,
            text: inline`
                       ├hello";┤
            `,
            replaceRange: {
              start: context.position,
              end: context.position,
            },
          },
        ],
      };
      const expected = {
        id: "",
        choices: [
          {
            index: 0,
            text: inline`
                       ├hello";┤
            `,
            replaceRange: {
              start: context.position,
              end: context.position + 1,
            },
          },
        ],
      };
      expect(await calculateReplaceRangeBySyntax(response, context)).to.deep.equal(expected);
    });

    it("should handle auto closing quotes", async () => {
      const context = {
        ...documentContext`
        let htmlMarkup = \`║\`
        `,
        language: "typescript",
      };
      const response = {
        id: "",
        choices: [
          {
            index: 0,
            text: inline`
                           ├<h1>\${message}</h1>\`;┤
            `,
            replaceRange: {
              start: context.position,
              end: context.position,
            },
          },
        ],
      };
      const expected = {
        id: "",
        choices: [
          {
            index: 0,
            text: inline`
                           ├<h1>\${message}</h1>\`;┤
            `,
            replaceRange: {
              start: context.position,
              end: context.position + 1,
            },
          },
        ],
      };
      expect(await calculateReplaceRangeBySyntax(response, context)).to.deep.equal(expected);
    });

    it("should handle multiple auto closing brackets", async () => {
      const context = {
        ...documentContext`
        process.on('data', (data) => {║})
        `,
        language: "typescript",
      };
      const response = {
        id: "",
        choices: [
          {
            index: 0,
            text: inline`
                                      ├
              console.log(data);
            });┤
            `,
            replaceRange: {
              start: context.position,
              end: context.position,
            },
          },
        ],
      };
      const expected = {
        id: "",
        choices: [
          {
            index: 0,
            text: inline`
                                      ├
              console.log(data);
            });┤
            `,
            replaceRange: {
              start: context.position,
              end: context.position + 2,
            },
          },
        ],
      };
      expect(await calculateReplaceRangeBySyntax(response, context)).to.deep.equal(expected);
    });

    it("should handle multiple auto closing brackets", async () => {
      const context = {
        ...documentContext`
        let mat: number[][][] = [[[║]]]
        `,
        language: "typescript",
      };
      const response = {
        id: "",
        choices: [
          {
            index: 0,
            text: inline`
                                   ├1, 2], [3, 4]], [[5, 6], [7, 8]]];┤
            `,
            replaceRange: {
              start: context.position,
              end: context.position,
            },
          },
        ],
      };
      const expected = {
        id: "",
        choices: [
          {
            index: 0,
            text: inline`
                                   ├1, 2], [3, 4]], [[5, 6], [7, 8]]];┤
            `,
            replaceRange: {
              start: context.position,
              end: context.position + 3,
            },
          },
        ],
      };
      expect(await calculateReplaceRangeBySyntax(response, context)).to.deep.equal(expected);
    });

    it("should handle the bad case of calculateReplaceRangeByBracketStack", async () => {
      const context = {
        ...documentContext`
      function clamp(n: number, max: number, min: number): number {
        return Math.max(Math.min(║);
      }
      `,
        language: "typescript",
      };
      const response = {
        id: "",
        choices: [
          {
            index: 0,
            text: inline`
                                 ├n, max), min┤
          `,
            replaceRange: {
              start: context.position,
              end: context.position,
            },
          },
        ],
      };
      const expected = {
        id: "",
        choices: [
          {
            index: 0,
            text: inline`
                                 ├n, max), min┤
          `,
            replaceRange: {
              start: context.position,
              end: context.position,
            },
          },
        ],
      };
      expect(await calculateReplaceRangeBySyntax(response, context)).to.deep.equal(expected);
    });
  });
});
