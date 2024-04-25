import { graphql } from '@/lib/gql/generates'

const updateGithubProvidedRepositoryActiveMutation = graphql(/* GraphQL */ `
  mutation UpdateGithubProvidedRepositoryActive($id: ID!, $active: Boolean!) {
    updateGithubProvidedRepositoryActive(id: $id, active: $active)
  }
`)

export { updateGithubProvidedRepositoryActiveMutation }
