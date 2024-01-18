import { TypedDocumentNode } from '@graphql-typed-document-node/core'
import { GraphQLClient, Variables } from 'graphql-request'
import { FieldValues, UseFormReturn } from 'react-hook-form'
import useSWR, { SWRConfiguration, SWRResponse } from 'swr'
import { useMutation as useUrqlMutation, AnyVariables, CombinedError } from 'urql'

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

export function useMutation<TResult, TVariables extends AnyVariables>(
  document: TypedDocumentNode<TResult, TVariables>,
  options?: {
    onCompleted?: (data: TResult) => void
    onError?: (err: CombinedError) => any
    form?: any
  }
) {
  const [mutationResult, executeMutation] = useUrqlMutation<TResult>(document)
  const onFormError = options?.form
    ? makeFormErrorHandler(options.form)
    : undefined

  const fn = async (variables?: TVariables) => {
    let res: TResult | undefined
    try {
      const response = await executeMutation(variables)
      
      if (response?.error) {
        onFormError && onFormError(response.error)
        options?.onError && options.onError(response.error)
        return
      }

      // todo not only return data?
      res = response?.data
    } catch (err: any) {
      options?.onError && options.onError(err)
      return
    }

    res && options?.onCompleted && options.onCompleted(res)
    return res
  }

  return fn
}

function makeFormErrorHandler<T extends FieldValues>(form: UseFormReturn<T>) {
  return (err: CombinedError) => {
    const { graphQLErrors = [] } = err
    for (const error of graphQLErrors) {
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
