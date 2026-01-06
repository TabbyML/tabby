import { graphql } from '@/lib/gql/generates'

export const updateIntegratedRepositoryActiveMutation = graphql(/* GraphQL */ `
  mutation UpdateIntegratedRepositoryActive(
    $id: ID!
    $active: Boolean!
    $refs: [String!]
  ) {
    updateIntegratedRepositoryActive(id: $id, active: $active, refs: $refs)
  }
`)

export const updateIntegratedRepositoryRefsMutation = graphql(/* GraphQL */ `
  mutation UpdateIntegratedRepositoryRefs($id: ID!, $refs: [String!]!) {
    updateIntegratedRepositoryRefs(id: $id, refs: $refs)
  }
`)
