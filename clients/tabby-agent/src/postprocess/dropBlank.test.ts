import { expect } from "chai";
import { dropBlank } from "./dropBlank";

describe("postprocess", () => {
  describe("dropBlank", () => {
    it("should return null if input is blank", () => {
      expect(dropBlank()("\n")).to.be.null;
      expect(dropBlank()("\t\n")).to.be.null;
    });
    it("should keep unchanged if input is not blank", () => {
      expect(dropBlank()("Not blank")).to.eq("Not blank");
    });
  });
});
