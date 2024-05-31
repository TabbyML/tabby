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

export const updateGithubSelfHostedProvidedRepositoryActiveMutation =
  graphql(/* GraphQL */ `
    mutation UpdateGithubSelfHostedProvidedRepositoryActive(
      $id: ID!
      $active: Boolean!
    ) {
      updateGithubSelfHostedProvidedRepositoryActive(id: $id, active: $active)
    }
  `)

export const updateGitlabSelfHostedProvidedRepositoryActiveMutation =
  graphql(/* GraphQL */ `
    mutation UpdateGitlabSelfHostedProvidedRepositoryActive(
      $id: ID!
      $active: Boolean!
    ) {
      updateGitlabSelfHostedProvidedRepositoryActive(id: $id, active: $active)
    }
  `)
