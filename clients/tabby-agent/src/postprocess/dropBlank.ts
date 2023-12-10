import { PostprocessFilter } from "./base";
import { isBlank } from "../utils";

export function dropBlank(): PostprocessFilter {
  return (input: string) => {
    return isBlank(input) ? null : input;
  };
}
