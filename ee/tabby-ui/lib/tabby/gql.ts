import { TypedDocumentNode } from '@graphql-typed-document-node/core'
import { GraphQLClient, Variables } from 'graphql-request'
import { GraphQLResponse } from 'graphql-request/build/esm/types'
import useSWR, { SWRConfiguration, SWRResponse } from 'swr'

export const gqlClient = new GraphQLClient(
  `${process.env.NEXT_PUBLIC_TABBY_SERVER_URL ?? ''}/graphql`
)

export interface ValidationError {
  path: string
  message: string
}

export interface ValidationErrors {
  errors: Array<ValidationError>
}

export function useGraphQLForm<
  TResult,
  TVariables extends Variables | undefined
>(
  document: TypedDocumentNode<TResult, TVariables>,
  options?: {
    onSuccess?: (values: TResult) => void
    onError?: (path: string, message: string) => void
  }
) {
  const onSubmit = async (values: TVariables) => {
    let res
    try {
      res = await gqlClient.request(document, values)
    } catch (err) {
      const { errors = [] } = (err as any).response as GraphQLResponse
      for (const error of errors) {
        if (error.extensions && error.extensions['validation-errors']) {
          const validationErrors = error.extensions[
            'validation-errors'
          ] as ValidationErrors
          for (const error of validationErrors.errors) {
            options?.onError && options?.onError(error.path, error.message)
          }
        } else {
          options?.onError && options?.onError('root', error.message)
        }
      }

      return res
    }

    options?.onSuccess && options.onSuccess(res)
  }
  return { onSubmit }
}

export function useGraphQLQuery<
  TResult,
  TVariables extends Variables | undefined
>(
  document: TypedDocumentNode<TResult, TVariables>,
  variables?: TVariables,
  swrConfiguration?: SWRConfiguration<TResult>
): SWRResponse<TResult> {
  return useSWR(
    [document, variables],
    ([document, variables]) => gqlClient.request(document, variables),
    swrConfiguration
  )
}
