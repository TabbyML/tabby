import { graphql } from '@/lib/gql/generates'

export const updateGithubProvidedRepositoryActiveMutation =
  graphql(/* GraphQL */ `
    mutation UpdateGithubProvidedRepositoryActive($id: ID!, $active: Boolean!) {
      updateGithubProvidedRepositoryActive(id: $id, active: $active)
    }
  `)

export const updateGitlabProvidedRepositoryActiveMutation =
  graphql(/* GraphQL */ `
    mutation UpdateGitlabProvidedRepositoryActive($id: ID!, $active: Boolean!) {
      updateGitlabProvidedRepositoryActive(id: $id, active: $active)
    }
  `)
