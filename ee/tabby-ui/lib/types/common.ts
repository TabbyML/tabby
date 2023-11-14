export type ServerActionResult<Result> = Promise<
  | Result
  | {
      error: string
    }
>
