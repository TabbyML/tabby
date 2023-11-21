import useSWR, { SWRResponse, SWRConfiguration } from 'swr'
import { request } from '@/lib/tabby-gql-client'
import { Variables } from 'graphql-request'
import { TypedDocumentNode } from '@graphql-typed-document-node/core'

function useGraphQL<TResult, TVariables extends Variables | undefined>(
  document: TypedDocumentNode<TResult, TVariables>,
  variables?: TVariables,
  swrConfiguration?: SWRConfiguration<TResult>
): SWRResponse<TResult> {
  return useSWR(
    [document, variables],
    ([document, variables]) => request({ document, variables }),
    swrConfiguration
  )
}

export { useGraphQL }
