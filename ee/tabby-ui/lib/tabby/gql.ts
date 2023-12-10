import { TypedDocumentNode } from '@graphql-typed-document-node/core'
import { GraphQLClient, Variables } from 'graphql-request'
import { GraphQLResponse } from 'graphql-request/build/esm/types'
import { FieldValues, UseFormReturn } from 'react-hook-form'
import useSWR, { SWRConfiguration, SWRResponse } from 'swr'

import { useSession } from './auth'

const gqlClient = new GraphQLClient(
  `${process.env.NEXT_PUBLIC_TABBY_SERVER_URL ?? ''}/graphql`
)

export interface ValidationError {
  path: string
  message: string
}

export interface ValidationErrors {
  errors: Array<ValidationError>
}

export function useMutation<TResult, TVariables extends Variables | undefined>(
  document: TypedDocumentNode<TResult, TVariables>,
  options?: {
    onCompleted?: (data: TResult) => void
    onError?: (err: any) => any
    form?: any
  }
) {
  const { data: session } = useSession()
  const onFormError = options?.form
    ? makeFormErrorHandler(options.form)
    : undefined

  const fn = async (variables?: TVariables) => {
    let res: TResult | undefined
    try {
      res = await gqlClient.request({
        document,
        variables: variables,
        requestHeaders: session
          ? {
              authorization: `Bearer ${session.accessToken}`
            }
          : undefined
      })
    } catch (err) {
      onFormError && onFormError(err)
      options?.onError && options.onError(err)
      return
    }

    options?.onCompleted && options.onCompleted(res)
  }

  return fn
}

function makeFormErrorHandler<T extends FieldValues>(form: UseFormReturn<T>) {
  return (err: any) => {
    const { errors = [] } = err.response as GraphQLResponse
    for (const error of errors) {
      if (error.extensions && error.extensions['validation-errors']) {
        const validationErrors = error.extensions[
          'validation-errors'
        ] as ValidationErrors
        for (const error of validationErrors.errors) {
          form.setError(error.path as any, error)
        }
      } else {
        form.setError('root', error)
      }
    }
  }
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
    ([document, variables]) =>
      gqlClient.request({
        document,
        variables
      }),
    swrConfiguration
  )
}

export function useAuthenticatedGraphQLQuery<
  TResult,
  TVariables extends Variables | undefined
>(
  document: TypedDocumentNode<TResult, TVariables>,
  variables?: TVariables,
  swrConfiguration?: SWRConfiguration<TResult>
): SWRResponse<TResult> {
  const { data, status } = useSession()
  return useSWR(
    status === 'authenticated'
      ? [document, variables, data?.accessToken]
      : null,
    ([document, variables, accessToken]) =>
      gqlClient.request({
        document,
        variables,
        requestHeaders: {
          authorization: `Bearer ${accessToken}`
        }
      }),
    swrConfiguration
  )
}
