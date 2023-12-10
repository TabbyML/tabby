import { expect } from "chai";
import { dropBlank } from "./dropBlank";
import { documentContext } from "./testUtils";

describe("postprocess", () => {
  describe("dropBlank", () => {
    const dummyContext = {
      ...documentContext`
      dummy║
      `,
      language: "plaintext",
    };

    it("should return null if input is blank", () => {
      expect(dropBlank()("\n", dummyContext)).to.be.null;
      expect(dropBlank()("\t\n", dummyContext)).to.be.null;
    });
    it("should keep unchanged if input is not blank", () => {
      expect(dropBlank()("Not blank", dummyContext)).to.eq("Not blank");
    });
  });
});
