import { CompletionRequest, CompletionResponse } from "../Agent";
import { rootLogger } from "../logger";

export type PostprocessContext = CompletionRequest;
export type PostprocessFilter = (item: string) => string | null | Promise<string | null>;

export const logger = rootLogger.child({ component: "Postprocess" });

export const applyFilter = (filter: PostprocessFilter) => {
  return async (response: CompletionResponse) => {
    response.choices = (
      await Promise.all(
        response.choices.map(async (choice) => {
          choice.text = await filter(choice.text);
          return choice;
        })
      )
    ).filter(Boolean);
    return response;
  };
};
