import { documentContext, inline, assertFilterResult, assertFilterResultNotEqual } from "./testUtils";
import { limitScopeByIndentation } from "./limitScopeByIndentation";

describe("postprocess", () => {
  describe("limitScopeByIndentation", () => {
    const filter = limitScopeByIndentation();
    it("should limit scope at sentence end, when completion is continuing uncompleted sentence in the prefix.", async () => {
      const context = documentContext`
        let a =║
      `;
      context.language = "javascript";
      const completion = inline`
               ├ 1;
        let b = 2;┤
      `;
      const expected = inline`
               ├ 1;┤
      `;
      await assertFilterResult(filter, context, completion, expected);
    });

    it("should limit scope at sentence end, when completion is continuing uncompleted sentence in the prefix.", async () => {
      const context = documentContext`
        function safeParse(json) {
          try {
            console.log║
          } catch (error) {
            console.error(error);
            return null;
          }
        }
      `;
      context.language = "javascript";
      const completion = inline`
                        ├("Parsing", { json });
            return JSON.parse(json);
          } catch (e) {
            return null;
          }
        }┤
      `;
      const expected = inline`
                        ├("Parsing", { json });┤
      `;
      await assertFilterResult(filter, context, completion, expected);
    });

    it("should limit scope at next indent level, including closing line, when completion is starting a new indent level in next line.", async () => {
      const context = documentContext`
        function findMax(arr) {║}
      `;
      context.language = "javascript";
      const completion = inline`
                               ├
          let max = arr[0];
          for (let i = 1; i < arr.length; i++) {
            if (arr[i] > max) {
              max = arr[i];
            }
          }
          return max;
        }
        console.log(findMax([1, 2, 3, 4, 5]));┤
      `;
      const expected = inline`
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
      await assertFilterResult(filter, context, completion, expected);
    });

    it("should limit scope at next indent level, including closing line, when completion is continuing uncompleted sentence in the prefix, and starting a new indent level in next line.", async () => {
      const context = documentContext`
        function findMax(arr) {
          let max = arr[0];
          for║
        }
      `;
      context.language = "javascript";
      const completion = inline`
             ├ (let i = 1; i < arr.length; i++) {
            if (arr[i] > max) {
              max = arr[i];
            }
          }
          return max;
        }
        console.log(findMax([1, 2, 3, 4, 5]));┤
      `;
      const expected = inline`
             ├ (let i = 1; i < arr.length; i++) {
            if (arr[i] > max) {
              max = arr[i];
            }
          }┤
        ┴┴
      `;
      await assertFilterResult(filter, context, completion, expected);
    });

    it("should limit scope at current indent level, including closing line, when completion starts new sentences at same indent level.", async () => {
      const context = documentContext`
        function findMax(arr) {
          let max = arr[0];║
        }
      `;
      context.language = "javascript";
      const completion = inline`
                           ├
          for (let i = 1; i < arr.length; i++) {
            if (arr[i] > max) {
              max = arr[i];
            }
          }
          return max;
        }┤
      `;
      const expected = completion;
      await assertFilterResult(filter, context, completion, expected);
    });

    it("should allow only one level closing bracket", async () => {
      const context = documentContext`
        function safeParse(json) {
          try {
            return JSON.parse(json);
          } catch (e) {
            return null;║
      `;
      context.language = "javascript";
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
      await assertFilterResult(filter, context, completion, expected);
    });

    it("should allow level closing bracket at current line, it looks same as starts new sentences", async () => {
      const context = documentContext`
        function helloworld() {
          console.log("hello");
          ║
      `;
      context.language = "javascript";
      const completion = inline`
          ├}┤
      `;
      const expected = completion;
      await assertFilterResult(filter, context, completion, expected);
    });

    it("should not allow level closing bracket, when the suffix lines have same indent level", async () => {
      const context = documentContext`
        function helloworld() {
          console.log("hello");║
          console.log("world");
        }
      `;
      context.language = "javascript";
      const completion = inline`
                               ├
        }┤
      `;
      const expected = inline`
                               ├┤`;
      await assertFilterResult(filter, context, completion, expected);
    });

    it("should use indent level of previous line, when current line is empty.", async () => {
      const context = documentContext`
        function safeParse(json) {
          try {
            ║
          }
        }
      `;
      context.language = "javascript";
      const completion = inline`
            ├return JSON.parse(json);
          } catch (e) {
            return null;
          }
        }┤
      `;
      const expected = inline`
            ├return JSON.parse(json);
          } catch (e) {
            return null;
          }┤
        ┴┴
      `;
      await assertFilterResult(filter, context, completion, expected);
    });
  });

  describe("limitScopeByIndentation: bad cases", () => {
    const filter = limitScopeByIndentation();
    it("cannot handle the case of indent that does'nt have a close line, e.g. chaining call", async () => {
      const context = documentContext`
        function sortWords(input) {
          const output = input.trim()
            .split("\n")
            .map((line) => line.split(" "))
            ║
        }
      `;
      context.language = "javascript";
      const completion = inline`
            ├.flat()
            .sort()
            .join(" ");
          console.log(output);
          return output;
        }
        sortWords("world hello");┤
      `;
      const expected = inline`
            ├.flat()
            .sort()
            .join(" ");
          console.log(output);
          return output;
        }┤
      `;
      await assertFilterResultNotEqual(filter, context, completion, expected);
    });

    it("cannot handle the case of indent that does'nt have a close line, e.g. python def function", async () => {
      const context = documentContext`
        def findMax(arr):
          ║
      `;
      context.language = "python";
      const completion = inline`
          ├max = arr[0]
          for i in range(1, len(arr)):
            if arr[i] > max:
              max = arr[i]
          return max
        findMax([1, 2, 3, 4, 5])┤
      `;
      const expected = inline`
          ├max = arr[0]
          for i in range(1, len(arr)):
            if arr[i] > max:
              max = arr[i]
          return max┤
      `;
      await assertFilterResultNotEqual(filter, context, completion, expected);
    });
  });
});
