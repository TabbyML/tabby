import { expect } from "chai";
import { documentContext, inline } from "./testUtils";
import { formatIndentation } from "./formatIndentation";

describe("postprocess", () => {
  describe("formatIndentation", () => {
    it("should format indentation if first line of completion is over indented.", () => {
      const context = {
        ...documentContext`
        function clamp(n: number, max: number, min: number): number {
          ║
        }
        `,
        indentation: "  ",
        language: "typescript",
      };
      const completion = inline`
          ├  return Math.max(Math.min(n, max), min);┤
      `;
      const expected = inline`
          ├return Math.max(Math.min(n, max), min);┤
      `;
      expect(formatIndentation()(completion, context)).to.eq(expected);
    });

    it("should format indentation if first line of completion is wrongly indented.", () => {
      const context = {
        ...documentContext`
        function clamp(n: number, max: number, min: number): number {
        ║
        }
        `,
        indentation: "    ",
        language: "typescript",
      };
      const completion = inline`
        ├  return Math.max(Math.min(n, max), min);┤
      `;
      const expected = inline`
        ├    return Math.max(Math.min(n, max), min);┤
      `;
      expect(formatIndentation()(completion, context)).to.eq(expected);
    });

    it("should format indentation if completion lines is over indented.", () => {
      const context = {
        ...documentContext`
        def findMax(arr):║
        `,
        indentation: "  ",
        language: "python",
      };
      const completion = inline`
                         ├
            max = arr[0]
            for i in range(1, len(arr)):
                if arr[i] > max:
                    max = arr[i]
            return max
        }┤
      `;
      const expected = inline`
                         ├
          max = arr[0]
          for i in range(1, len(arr)):
            if arr[i] > max:
              max = arr[i]
          return max
        }┤
      `;
      expect(formatIndentation()(completion, context)).to.eq(expected);
    });

    it("should format indentation if completion lines is wrongly indented.", () => {
      const context = {
        ...documentContext`
        def findMax(arr):║
        `,
        indentation: "    ",
        language: "python",
      };
      const completion = inline`
                         ├
          max = arr[0]
          for i in range(1, len(arr)):
            if arr[i] > max:
              max = arr[i]
          return max
        }┤
      `;
      const expected = inline`
                         ├
            max = arr[0]
            for i in range(1, len(arr)):
                if arr[i] > max:
                    max = arr[i]
            return max
        }┤
      `;
      expect(formatIndentation()(completion, context)).to.eq(expected);
    });

    it("should keep it unchanged if it no indentation specified.", () => {
      const context = {
        ...documentContext`
        def findMax(arr):║
        `,
        indentation: undefined,
        language: "python",
      };
      const completion = inline`
                          ├
            max = arr[0]
            for i in range(1, len(arr)):
                if arr[i] > max:
                    max = arr[i]
            return max
        }┤
      `;
      expect(formatIndentation()(completion, context)).to.eq(completion);
    });

    it("should keep it unchanged if there is indentation in the context.", () => {
      const context = {
        ...documentContext`
        def hello():
            return "world"

        def findMax(arr):║
        `,
        indentation: "\t",
        language: "python",
      };
      const completion = inline`
                          ├
            max = arr[0]
            for i in range(1, len(arr)):
                if arr[i] > max:
                    max = arr[i]
            return max
        }┤
      `;
      expect(formatIndentation()(completion, context)).to.eq(completion);
    });

    it("should keep it unchanged if it is well indented.", () => {
      const context = {
        ...documentContext`
        def findMax(arr):║
        `,
        indentation: "    ",
        language: "python",
      };
      const completion = inline`
                          ├
            max = arr[0]
            for i in range(1, len(arr)):
                if arr[i] > max:
                    max = arr[i]
            return max
        }┤
      `;
      expect(formatIndentation()(completion, context)).to.eq(completion);
    });
  });
});
