import { expect } from "chai";
import { documentContext, inline } from "./testUtils";
import { removeDuplicatedLineSuffix } from "./removeDuplicatedLineSuffix";

describe("postprocess", () => {
  describe("removeDuplicatedLineSuffix", () => {
    it("should remove duplicated line suffix", () => {
      const context = {
        ...documentContext`
        const log = ({ ok }: { ok:║ }) => {
          console.log(ok);
        } 
        `,
        language: "javascript",
      };
      const completion = inline`├ boolean }) => {┤`;

      const expected = inline`├ boolean┤`;
      expect(removeDuplicatedLineSuffix()(completion, context)).to.be.equal(expected);
    });
  });
});
