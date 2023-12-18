import { expect } from "chai";
import { documentContext, inline } from "./testUtils";
import { dropMultiLineInSingleLineMode } from "./dropMultiLineInSingleLineMode";

describe("postprocess", () => {
  describe("dropMultiLineInSingleLineMode", () => {
    it("should drop multiline completions, when the suffix have non-auto-closed chars in the current line.", () => {
      const context = {
        ...documentContext`
        let error = new Error("Something went wrong");
        console.log(║message);
        `,
        language: "javascript",
      };
      const completion = inline`
                    ├message);
        throw error;┤
      `;
      expect(dropMultiLineInSingleLineMode()(completion, context)).to.be.null;
    });

    it("should allow singleline completions, when the suffix have non-auto-closed chars in the current line.", () => {
      const context = {
        ...documentContext`
        let error = new Error("Something went wrong");
        console.log(║message);
        `,
        language: "javascript",
      };
      const completion = inline`
                    ├error, ┤
      `;
      expect(dropMultiLineInSingleLineMode()(completion, context)).to.eq(completion);
    });

    it("should allow multiline completions, when the suffix only have auto-closed chars that will be replaced in the current line, such as `)]}`.", () => {
      const context = {
        ...documentContext`
        function findMax(arr) {║}
        `,
        language: "javascript",
      };
      const completion = inline`
                               ├
          let max = arr[0];
          for (let i = 1; i < arr.length; i++) {
            if (arr[i] > max) {
              max = arr[i];
            }
          }
          return max;
        }┤
      `;
      expect(dropMultiLineInSingleLineMode()(completion, context)).to.eq(completion);
    });
  });
});
