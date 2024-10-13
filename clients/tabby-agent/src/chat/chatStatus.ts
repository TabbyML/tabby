import { Edit } from "./inlineEdit";

export class ChatStatus {
  public static currentEdit: Edit | undefined = undefined;
  public static mutexAbortController: AbortController | undefined = undefined;
}
