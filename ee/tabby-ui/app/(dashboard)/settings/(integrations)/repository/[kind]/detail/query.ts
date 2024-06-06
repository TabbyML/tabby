import { graphql } from '@/lib/gql/generates'

export const updateIntegratedRepositoryActiveMutation = graphql(/* GraphQL */ `
  mutation UpdateIntegratedRepositoryActive($id: ID!, $active: Boolean!) {
    updateIntegratedRepositoryActive(id: $id, active: $active)
  }
`)
