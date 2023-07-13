import { PostprocessFilter } from "./base";
import { isBlank } from "../utils";

export const dropBlank: () => PostprocessFilter = () => {
  return (input) => {
    return isBlank(input) ? null : input;
  };
};
