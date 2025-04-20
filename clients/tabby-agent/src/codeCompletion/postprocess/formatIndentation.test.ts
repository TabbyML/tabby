import { documentContext, inline, assertFilterResult } from "./testUtils";
import { formatIndentation } from "./formatIndentation";

describe("postprocess", () => {
  describe("formatIndentation", () => {
    const filter = formatIndentation();
    it("should format indentation if first line of completion is over indented.", async () => {
      const context = documentContext`typescript
        function clamp(n: number, max: number, min: number): number {
          ║
        }
      `;
      const indentation = "  ";
      const completion = inline`
          ├  return Math.max(Math.min(n, max), min);┤
      `;
      const expected = inline`
          ├return Math.max(Math.min(n, max), min);┤
      `;
      await assertFilterResult(filter, { ...context, editorOptions: { indentation } }, completion, expected);
    });

    it("should format indentation if first line of completion is wrongly indented.", async () => {
      const context = documentContext`typescript
        function clamp(n: number, max: number, min: number): number {
        ║
        }
      `;
      const indentation = "    ";
      const completion = inline`
        ├  return Math.max(Math.min(n, max), min);┤
      `;
      const expected = inline`
        ├    return Math.max(Math.min(n, max), min);┤
      `;
      await assertFilterResult(filter, { ...context, editorOptions: { indentation } }, completion, expected);
    });

    it("should format indentation if completion lines is over indented.", async () => {
      const context = documentContext`python
        def findMax(arr):║
      `;
      const indentation = "  ";
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
      await assertFilterResult(filter, { ...context, editorOptions: { indentation } }, completion, expected);
    });

    it("should format indentation if completion lines is wrongly indented.", async () => {
      const context = documentContext`python
        def findMax(arr):║
      `;
      const indentation = "    ";
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
      await assertFilterResult(filter, { ...context, editorOptions: { indentation } }, completion, expected);
    });

    it("should keep it unchanged if it no indentation specified.", async () => {
      const context = documentContext`python
        def findMax(arr):║
      `;
      const indentation = undefined;
      const completion = inline`
                          ├
            max = arr[0]
            for i in range(1, len(arr)):
                if arr[i] > max:
                    max = arr[i]
            return max
        }┤
      `;
      const expected = completion;
      await assertFilterResult(filter, { ...context, editorOptions: indentation }, completion, expected);
    });

    it("should keep it unchanged if there is indentation in the context.", async () => {
      const context = documentContext`python
        def hello():
            return "world"

        def findMax(arr):║
      `;
      const indentation = "\t";
      const completion = inline`
                          ├
            max = arr[0]
            for i in range(1, len(arr)):
                if arr[i] > max:
                    max = arr[i]
            return max
        }┤
      `;
      const expected = completion;
      await assertFilterResult(filter, { ...context, editorOptions: { indentation } }, completion, expected);
    });

    it("should keep it unchanged if it is well indented.", async () => {
      const context = documentContext`python
        def findMax(arr):║
      `;
      const indentation = "    ";
      const completion = inline`
                          ├
            max = arr[0]
            for i in range(1, len(arr)):
                if arr[i] > max:
                    max = arr[i]
            return max
        }┤
      `;
      const expected = completion;
      await assertFilterResult(filter, { ...context, editorOptions: { indentation } }, completion, expected);
    });
  });
});
