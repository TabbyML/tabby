import { graphql } from '@/lib/gql/generates'

export const updateIntegratedRepositoryActiveMutation = graphql(/* GraphQL */ `
  mutation UpdateIntegratedRepositoryActive($id: ID!, $active: Boolean!) {
    updateIntegratedRepositoryActive(id: $id, active: $active)
  }
`)

export const updateIntegratedRepositoryRefsMutation = graphql(/* GraphQL */ `
  mutation UpdateIntegratedRepositoryRefs($id: ID!, $refs: [String!]!) {
    updateIntegratedRepositoryRefs(id: $id, refs: $refs)
  }
`)
