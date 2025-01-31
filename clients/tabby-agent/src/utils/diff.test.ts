import { Range } from "vscode-languageserver";
import { calcCharDiffRange } from "./diff";
import { expect } from "chai";

describe("diff", () => {
  describe("calcCharDiffRange", () => {
    it("diff chars test case 1", () => {
      /**
       *  <<<<<<< tabby-000000
       *  Deleted Line
       *  Unchanged Line
       *  Added Line
       *  Modified Line before
       *  Modified Line after
       *  >>>>>>> tabby-000000 [-=+-+]
       */
      const originText = `Deleted Line
Unchanged Line
Modified Line (Before Changes)`;
      const editText = `Unchanged Line
Added Line
Modified Line (After Changes)`;
      const editTextRanges: Range[] = [
        { start: { line: 2, character: 0 }, end: { line: 2, character: 14 } },
        { start: { line: 3, character: 0 }, end: { line: 3, character: 10 } },
        { start: { line: 5, character: 0 }, end: { line: 5, character: 29 } },
      ];
      const ranges = calcCharDiffRange(originText, editText, editTextRanges);
      const diffRanges = [
        { start: { line: 2, character: 0 }, end: { line: 2, character: 7 } },
        { start: { line: 3, character: 0 }, end: { line: 3, character: 3 } },
        { start: { line: 5, character: 15 }, end: { line: 5, character: 18 } },
      ];
      expect(ranges).to.deep.equal(diffRanges);
    });

    it("diff chars test case 2", () => {
      /**
       *  <<<<<<< tabby-8td8Ip
       * examples/assets/downloads/*
       * !examples/assets/downloads/.tracked
       * examples/headless/outputs/*
       * !examples/headless/outputs/.tracked
       *
       * examples/Assets/downloads/*
       * !examples/Assets/downloads/.tracked
       * examples/Headless/outputs/*
       * !examples/Headless/outputs/.tracked
       * >>>>>>> tabby-8td8Ip [----+++++]
       */
      const originText = `examples/assets/downloads/*
!examples/assets/downloads/.tracked
examples/headless/outputs/*
!examples/headless/outputs/.tracked`;
      const editText = `
examples/Assets/downloads/*
!examples/Assets/downloads/.tracked
examples/Headless/outputs/*
!examples/Headless/outputs/.tracked`;
      const editTextRanges: Range[] = [
        { start: { line: 9, character: 0 }, end: { line: 9, character: 0 } },
        { start: { line: 10, character: 0 }, end: { line: 10, character: 27 } },
        { start: { line: 11, character: 0 }, end: { line: 11, character: 35 } },
        { start: { line: 12, character: 0 }, end: { line: 12, character: 27 } },
        { start: { line: 13, character: 0 }, end: { line: 13, character: 35 } },
      ];
      const ranges = calcCharDiffRange(originText, editText, editTextRanges);
      const diffRanges = [
        { start: { line: 9, character: 0 }, end: { line: 9, character: 1 } },
        { start: { line: 10, character: 9 }, end: { line: 10, character: 10 } },
        { start: { line: 11, character: 10 }, end: { line: 11, character: 11 } },
        { start: { line: 12, character: 9 }, end: { line: 12, character: 10 } },
        { start: { line: 13, character: 10 }, end: { line: 13, character: 11 } },
      ];
      expect(ranges).to.deep.equal(diffRanges);
    });
  });
});
