import { GraphQLError } from 'graphql'
import { CombinedError } from 'urql'

export type ServerActionResult<Result> = Promise<
  | Result
  | {
      error: string
    }
>

// get element type from array type
export type ArrayElementType<T> = T extends Array<infer T> ? T : never

export interface ExtendedCombinedError
  extends Omit<CombinedError, 'graphQLErrors'> {
  graphQLErrors?: GraphQLError[]
}
