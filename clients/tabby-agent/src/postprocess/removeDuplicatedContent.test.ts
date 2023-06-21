import { expect } from "chai";
import dedent from "dedent";
import { removeDuplicatedContent } from "./removeDuplicatedContent";

const buildContext = (doc: string) => {
  return {
    filepath: null,
    language: "javascript",
    text: doc.replace(/║/, ""),
    position: doc.indexOf("║"),
  };
};

describe("postprocess", () => {
  describe("removeDuplicatedContent", () => {
    it("should remove duplicated content", () => {
      const context = buildContext(dedent`
        function sum(a, b) {
          ║
          return value;
        }
      `);
      const completion = dedent`
          let value = a + b;
          return value;
        }
      `;
      const expected = dedent`
          let value = a + b;
      `;
      expect(removeDuplicatedContent(context)(completion)).to.eq(expected);
    });

    it("can not remove similar but different content for now", () => {
      const context = buildContext(dedent`
        let sum = (a, b) => {
          ║return a + b;
        }
      `);
      const completion = dedent`
          return a + b;
        };
      `;
      expect(removeDuplicatedContent(context)(completion)).to.eq(completion);
    });
  });
});
