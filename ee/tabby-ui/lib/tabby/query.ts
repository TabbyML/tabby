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
  query gitRepositories(
    $after: String
    $before: String
    $first: Int
    $last: Int
  ) {
    gitRepositories(
      after: $after
      before: $before
      first: $first
      last: $last
    ) {
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
    $ids: [ID!]
    $jobs: [String!]
    $after: String
    $before: String
    $first: Int
    $last: Int
  ) {
    jobRuns(
      ids: $ids
      jobs: $jobs
      after: $after
      before: $before
      first: $first
      last: $last
    ) {
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

export const queryJobRunStats = graphql(/* GraphQL */ `
  query GetJobRunStats($jobs: [String!]) {
    jobRunStats(jobs: $jobs) {
      success
      failed
      pending
    }
  }
`)

export const listJobs = graphql(/* GraphQL */ `
  query ListJobs {
    jobs
  }
`)

export const listUsers = graphql(/* GraphQL */ `
  query ListUsers($after: String, $before: String, $first: Int, $last: Int) {
    users(after: $after, before: $before, first: $first, last: $last) {
      edges {
        node {
          id
          email
          isAdmin
          isOwner
          createdAt
          active
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

export const queryDailyStatsInPastYear = graphql(/* GraphQL */ `
  query DailyStatsInPastYear($users: [ID!]) {
    dailyStatsInPastYear(users: $users) {
      start
      end
      completions
      selects
      views
    }
  }
`)

export const queryDailyStats = graphql(/* GraphQL */ `
  query DailyStats(
    $start: DateTime!
    $end: DateTime!
    $users: [ID!]
    $languages: [Language!]
  ) {
    dailyStats(start: $start, end: $end, users: $users, languages: $languages) {
      start
      end
      completions
      selects
      views
      language
    }
  }
`)

export const listGithubRepositoryProviders = graphql(/* GraphQL */ `
  query ListGithubRepositoryProviders(
    $ids: [ID!]
    $after: String
    $before: String
    $first: Int
    $last: Int
  ) {
    githubRepositoryProviders(
      ids: $ids
      after: $after
      before: $before
      first: $first
      last: $last
    ) {
      edges {
        node {
          id
          displayName
          status
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

export const listGithubRepositories = graphql(/* GraphQL */ `
  query ListGithubRepositories(
    $providerIds: [ID!]!
    $active: Boolean
    $after: String
    $before: String
    $first: Int
    $last: Int
  ) {
    githubRepositories(
      providerIds: $providerIds
      active: $active
      after: $after
      before: $before
      first: $first
      last: $last
    ) {
      edges {
        node {
          id
          vendorId
          githubRepositoryProviderId
          name
          gitUrl
          active
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

export const listGitlabRepositoryProviders = graphql(/* GraphQL */ `
  query ListGitlabRepositoryProviders(
    $ids: [ID!]
    $after: String
    $before: String
    $first: Int
    $last: Int
  ) {
    gitlabRepositoryProviders(
      ids: $ids
      after: $after
      before: $before
      first: $first
      last: $last
    ) {
      edges {
        node {
          id
          displayName
          status
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

export const listGitlabRepositories = graphql(/* GraphQL */ `
  query ListGitlabRepositories(
    $providerIds: [ID!]!
    $active: Boolean
    $after: String
    $before: String
    $first: Int
    $last: Int
  ) {
    gitlabRepositories(
      providerIds: $providerIds
      active: $active
      after: $after
      before: $before
      first: $first
      last: $last
    ) {
      edges {
        node {
          id
          vendorId
          gitlabRepositoryProviderId
          name
          gitUrl
          active
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
