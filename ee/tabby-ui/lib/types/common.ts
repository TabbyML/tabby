export type ServerActionResult<Result> = Promise<
  | Result
  | {
      error: string
    }
>

// get element type from array type
export type ArrayElementType<T> = T extends Array<infer T> ? T : never
