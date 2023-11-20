import useSWR, { SWRResponse, SWRConfiguration } from 'swr'
import { request } from '@/lib/tabby-gql-client'
import { Variables } from 'graphql-request'
import { TypedDocumentNode } from '@graphql-typed-document-node/core'
import { ASTNode, Kind, OperationDefinitionNode } from 'graphql'

const isOperationDefinition = (def: ASTNode): def is OperationDefinitionNode =>
  def.kind === Kind.OPERATION_DEFINITION

function useGraphQL<TResult, TVariables extends Variables | undefined>(
  document: TypedDocumentNode<TResult, TVariables>,
  variables?: TVariables,
  options?: SWRConfiguration<TResult>
): SWRResponse<TResult> {
  return useSWR(
    [
      document.definitions.find(isOperationDefinition)?.name?.value,
      document,
      variables
    ],
    ([_key, document, variables]) => {
      return request({ document, variables })
    },
    options
  )
}

export { useGraphQL }
