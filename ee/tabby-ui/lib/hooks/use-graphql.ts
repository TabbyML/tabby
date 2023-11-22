import useSWR, { SWRResponse, SWRConfiguration } from 'swr'
import { gqlClient } from '@/lib/tabby-gql-client'
import { Variables } from 'graphql-request'
import { TypedDocumentNode } from '@graphql-typed-document-node/core'

function useGraphQL<TResult, TVariables extends Variables | undefined>(
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

export { useGraphQL }
