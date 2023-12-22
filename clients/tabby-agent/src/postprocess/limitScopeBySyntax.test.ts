import { expect } from "chai";
import { documentContext, inline } from "./testUtils";
import { limitScopeBySyntax } from "./limitScopeBySyntax";

describe("postprocess", () => {
  describe("limitScopeBySyntax javascript", () => {
    it("should limit scope at function_declaration.", async () => {
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
      expect(await limitScopeBySyntax()(completion, context)).to.eq(expected);
    });

    it("should limit scope at function_declaration", async () => {
      const context = {
        ...documentContext`
        function findMax(arr) {
          let max = arr[0];║
        }
        `,
        language: "javascript",
      };
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
      expect(await limitScopeBySyntax()(completion, context)).to.eq(completion);
    });

    it("should limit scope at function_declaration", async () => {
      const context = {
        ...documentContext`
        function findMax(arr) {
          let max = arr[0];
          for (let i = 1; i < arr.length; i++) {
            if (arr[i] > max) {
              max = arr[i];
            }
          }║
        }
        `,
        language: "javascript",
      };
      const completion = inline`
           ├
          return max;
        }┤
      `;
      expect(await limitScopeBySyntax()(completion, context)).to.eq(completion);
    });

    it("should limit scope at for_statement.", async () => {
      const context = {
        ...documentContext`
        function findMax(arr) {
          let max = arr[0];
          for║
        }
        `,
        language: "javascript",
      };
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
      expect(await limitScopeBySyntax()(completion, context)).to.eq(expected);
    });

    it("should limit scope at current node if no parent scope found.", async () => {
      const context = {
        ...documentContext`
        let a =║
        `,
        language: "javascript",
      };
      const completion = inline`
               ├ 1;
        let b = 2;┤
      `;
      const expected = inline`
               ├ 1;┤
      `;
      expect(await limitScopeBySyntax()(completion, context)).to.eq(expected);
    });

    it("should handle the bad case of limitScopeByIndentation", async () => {
      const context = {
        ...documentContext`
        function sortWords(input) {
          const output = input.trim()
            .split("\n")
            .map((line) => line.split(" "))
            ║
        }
        `,
        language: "javascript",
      };
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
      expect(await limitScopeBySyntax()(completion, context)).to.eq(expected);
    });
  });

  describe("limitScopeBySyntax python", () => {
    it("should limit scope at function_definition.", async () => {
      const context = {
        ...documentContext`
        def find_min(arr):║
        `,
        language: "python",
      };
      const completion = inline`
                          ├
          min = arr[0]
          for i in range(1, len(arr)):
            if arr[i] < min:
              min = arr[i]
          return min
        print(find_min([1, 2, 3, 4, 5]))┤
      `;
      const expected = inline`
                          ├
          min = arr[0]
          for i in range(1, len(arr)):
            if arr[i] < min:
              min = arr[i]
          return min┤
        ┴┴
      `;
      expect(await limitScopeBySyntax()(completion, context)).to.eq(expected);
    });

    it("should limit scope at function_definition.", async () => {
      const context = {
        ...documentContext`
        def find_min(arr):
          min = arr[0]║
        `,
        language: "python",
      };
      const completion = inline`
                      ├
          for i in range(1, len(arr)):
            if arr[i] < min:
              min = arr[i]
          return min
        print(find_min([1, 2, 3, 4, 5]))┤
      `;
      const expected = inline`
                      ├
          for i in range(1, len(arr)):
            if arr[i] < min:
              min = arr[i]
          return min┤
        ┴┴
      `;
      expect(await limitScopeBySyntax()(completion, context)).to.eq(expected);
    });

    it("should limit scope at function_definition.", async () => {
      const context = {
        ...documentContext`
        def find_min(arr):
          min = arr[0]
          for i in range(1, len(arr)):
            if arr[i] < min:
              min = arr[i]║
        `,
        language: "python",
      };
      const completion = inline`
                          ├
          return min┤
      `;
      expect(await limitScopeBySyntax()(completion, context)).to.eq(completion);
    });

    it("should limit scope at for_statement.", async () => {
      const context = {
        ...documentContext`
        def find_min(arr):
          max = arr[0]
          for║
        `,
        language: "python",
      };
      const completion = inline`
             ├ i in range(1, len(arr)):
            if arr[i] < min:
              min = arr[i]
          return min
        ┤
      `;
      const expected = inline`
             ├ i in range(1, len(arr)):
            if arr[i] < min:
              min = arr[i]┤
        ┴┴┴┴
      `;
      expect(await limitScopeBySyntax()(completion, context)).to.eq(expected);
    });

    it("should handle the bad case of limitScopeByIndentation", async () => {
      const context = {
        ...documentContext`
        def findMax(arr):
          ║
        `,
        language: "python",
      };
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
        ┴┴
      `;
      expect(await limitScopeBySyntax()(completion, context)).to.eq(expected);
    });
  });

  describe("limitScopeBySyntax go", () => {
    it("should limit scope at function_declaration.", async () => {
      const context = {
        ...documentContext`
        func findMin(arr []int) int {║}
        `,
        language: "go",
      };
      const completion = inline`
                                     ├
          min := math.MaxInt64
          for _, v := range arr {
            if v < min {
              min = v
            }
          }
          return min
        }
        
        func main() {
          arr := []int{5, 2, 9, 8, 1, 3}
          fmt.Println(findMin(arr)) // Output: 1
        }┤
      `;
      const expected = inline`
                                     ├
          min := math.MaxInt64
          for _, v := range arr {
            if v < min {
              min = v
            }
          }
          return min
        }┤
      `;
      expect(await limitScopeBySyntax()(completion, context)).to.eq(expected);
    });

    it("should limit scope at for_statement.", async () => {
      const context = {
        ...documentContext`
        func findMin(arr []int) int {
          min := math.MaxInt64
          for║
        `,
        language: "go",
      };
      const completion = inline`
             ├ _, v := range arr {
            if v < min {
              min = v
            }
          }
          return min
        }┤
      `;
      const expected = inline`
             ├ _, v := range arr {
            if v < min {
              min = v
            }
          }┤
        ┴┴
      `;
      expect(await limitScopeBySyntax()(completion, context)).to.eq(expected);
    });
  });

  describe("limitScopeBySyntax rust", () => {
    it("should limit scope at function_item.", async () => {
      const context = {
        ...documentContext`
        fn find_min(arr: &[i32]) -> i32 {║}
        `,
        language: "rust",
      };
      const completion = inline`
                                         ├
          *arr.iter().min().unwrap()
        }
        fn main() {
          let arr = vec![5, 2, 9, 8, 1, 3];
          println!("{}", find_min(&arr)); // Output: 1
        }┤
      `;
      const expected = inline`
                                         ├
          *arr.iter().min().unwrap()
        }┤
      `;
      expect(await limitScopeBySyntax()(completion, context)).to.eq(expected);
    });
  });

  describe("limitScopeBySyntax ruby", () => {
    it("should limit scope at for.", async () => {
      const context = {
        ...documentContext`
        def fibonacci(n)║
        `,
        language: "ruby",
      };
      const completion = inline`
                        ├
          return n if n <= 1
          fibonacci(n - 1) + fibonacci(n - 2)
        end
        puts fibonacci(10)┤
      `;
      const expected = inline`
                        ├
          return n if n <= 1
          fibonacci(n - 1) + fibonacci(n - 2)
        end┤
      `;
      expect(await limitScopeBySyntax()(completion, context)).to.eq(expected);
    });
  });
});
