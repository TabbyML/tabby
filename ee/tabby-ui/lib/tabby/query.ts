import { graphql } from '@/lib/gql/generates'

export const listInvitations = graphql(/* GraphQL */ `
  query ListInvitations(
    $after: String
    $before: String
    $first: Int
    $last: Int
  ) {
    invitations(after: $after, before: $before, first: $first, last: $last) {
      edges {
        node {
          id
          email
          code
          createdAt
        }
        cursor
      }
      pageInfo {
        hasNextPage
        hasPreviousPage
        startCursor
        endCursor
      }
    }
  }
`)

export const listRepositories = graphql(/* GraphQL */ `
  query repositories($after: String, $before: String, $first: Int, $last: Int) {
    repositories(after: $after, before: $before, first: $first, last: $last) {
      edges {
        node {
          id
          name
          gitUrl
        }
        cursor
      }
      pageInfo {
        hasNextPage
        hasPreviousPage
        startCursor
        endCursor
      }
    }
  }
`)

export const listJobRuns = graphql(/* GraphQL */ `
  query ListJobRuns(
    $after: String
    $before: String
    $first: Int
    $last: Int
  ) {
    jobRuns(after: $after, before: $before, first: $first, last: $last) {
      edges {
        node {
          id
          job
          createdAt
          finishedAt
          exitCode
          stdout
          stderr
        }
        cursor
      }
      pageInfo {
        hasNextPage
        hasPreviousPage
        startCursor
        endCursor
      }
    }
  }
`)
