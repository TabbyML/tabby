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
          jobInfo {
            lastJobRun {
              id
              job
              createdAt
              finishedAt
              exitCode
            }
            command
          }
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
          startedAt
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
          name
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

export const listIntegrations = graphql(/* GraphQL */ `
  query ListIntegrations(
    $ids: [ID!]
    $kind: IntegrationKind
    $after: String
    $before: String
    $first: Int
    $last: Int
  ) {
    integrations(
      ids: $ids
      kind: $kind
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
          apiBase
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

export const listIntegratedRepositories = graphql(/* GraphQL */ `
  query ListIntegratedRepositories(
    $ids: [ID!]
    $kind: IntegrationKind
    $active: Boolean
    $after: String
    $before: String
    $first: Int
    $last: Int
  ) {
    integratedRepositories(
      ids: $ids
      kind: $kind
      active: $active
      after: $after
      before: $before
      first: $first
      last: $last
    ) {
      edges {
        node {
          id
          displayName
          gitUrl
          active
          jobInfo {
            lastJobRun {
              id
              job
              createdAt
              finishedAt
              startedAt
              exitCode
            }
            command
          }
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

export const listWebCrawlerUrl = graphql(/* GraphQL */ `
  query WebCrawlerUrls(
    $after: String
    $before: String
    $first: Int
    $last: Int
  ) {
    webCrawlerUrls(after: $after, before: $before, first: $first, last: $last) {
      edges {
        node {
          url
          id
          createdAt
          jobInfo {
            lastJobRun {
              id
              job
              createdAt
              finishedAt
              exitCode
            }
            command
          }
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
