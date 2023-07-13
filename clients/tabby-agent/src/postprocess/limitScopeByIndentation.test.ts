import { expect } from "chai";
import { documentContext, inline } from "./testUtils";
import { limitScopeByIndentation } from "./limitScopeByIndentation";

describe("postprocess", () => {
  describe("limitScopeByIndentation", () => {
    it("should remove content out of current intent scope", () => {
      const context = {
        ...documentContext`
        function safeParse(json) {
          try {
            console.log║
          } catch (error) {
            console.error(error);
            return null;
          }
        }
        `,
        language: "javascript",
      };
      const completion = inline`
                        ├("Parsing", { json });
            return JSON.parse(json);
          } catch (e) {
            return null;
          }
        }┤
      `;
      const expected = inline`
                        ├("Parsing", { json });
            return JSON.parse(json);┤
        ┴┴┴┴
      `;
      expect(limitScopeByIndentation(context)(completion)).to.eq(expected);
    });

    it("should allow single level closing bracket", () => {
      const context = {
        ...documentContext`
        function safeParse(json) {
          try {
            return JSON.parse(json);
          } catch (e) {
            return null;║
        `,
        language: "javascript",
      };
      const completion = inline`
                        ├
          }
        }┤
      `;
      const expected = inline`
                        ├
          }┤
        ┴┴
      `;
      expect(limitScopeByIndentation(context)(completion)).to.eq(expected);
    });

    it("should allow single level closing bracket", () => {
      const context = {
        ...documentContext`
        function safeParse(json) {
          try {
            return JSON.parse(json);
          } catch (e) {
            ║
          }
        }`,
        language: "javascript",
      };
      const completion = inline`
            ├return null;
          }
        }┤
      `;
      // In fact, we do not expect the closing `}`, because of there are `}` already in suffix,
      // but we leave them here and pass to other filters to handle it.
      const expected = inline`
            ├return null;
          }┤
        ┴┴
      `;
      expect(limitScopeByIndentation(context)(completion)).to.eq(expected);
    });

    // Might be better to allow
    it("not allow back step indent level with `catch` or `else` if there is no similar block yet", () => {
      const context = {
        ...documentContext`
        function safeParse(json) {
          try {
            ║
          }
        }
        `,
        language: "javascript",
      };
      const completion = inline`
            ├return JSON.parse(json);
          } catch (e) {
            return null;
          }
        }┤
      `;
      const expected = inline`
            ├return JSON.parse(json);┤
        ┴┴┴┴
      `;
      expect(limitScopeByIndentation(context)(completion)).to.eq(expected);
    });
  });
});
