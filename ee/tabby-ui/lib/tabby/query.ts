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
          sourceId
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

export const listSecuredUsers = graphql(/* GraphQL */ `
  query ListUsers(
    $ids: [ID!]
    $after: String
    $before: String
    $first: Int
    $last: Int
  ) {
    users(
      ids: $ids
      after: $after
      before: $before
      first: $first
      last: $last
    ) {
      edges {
        node {
          id
          email
          isAdmin
          isOwner
          createdAt
          active
          name
          isSsoUser
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
          message
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
          sourceId
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

export const repositoryListQuery = graphql(/* GraphQL */ `
  query RepositoryList {
    repositoryList {
      id
      name
      kind
      gitUrl
      refs {
        name
        commit
      }
    }
  }
`)

export const repositorySearch = graphql(/* GraphQL */ `
  query RepositorySearch(
    $kind: RepositoryKind!
    $id: ID!
    $rev: String
    $pattern: String!
  ) {
    repositorySearch(kind: $kind, id: $id, rev: $rev, pattern: $pattern) {
      type
      path
      indices
    }
  }
`)

export const contextInfoQuery = graphql(/* GraphQL */ `
  query ContextInfo {
    contextInfo {
      sources {
        id
        sourceKind
        sourceId
        sourceName
      }
    }
  }
`)

export const userGroupsQuery = graphql(/* GraphQL */ `
  query UserGroups {
    userGroups {
      id
      name
      createdAt
      updatedAt
      members {
        user {
          id
          email
          name
          createdAt
        }
        isGroupAdmin
        createdAt
        updatedAt
      }
    }
  }
`)

export const listSourceIdAccessPolicies = graphql(/* GraphQL */ `
  query sourceIdAccessPolicies($sourceId: String!) {
    sourceIdAccessPolicies(sourceId: $sourceId) {
      sourceId
      read {
        id
        name
      }
    }
  }
`)

export const listThreads = graphql(/* GraphQL */ `
  query ListThreads(
    $ids: [ID!]
    $isEphemeral: Boolean
    $after: String
    $before: String
    $first: Int
    $last: Int
  ) {
    threads(
      ids: $ids
      isEphemeral: $isEphemeral
      after: $after
      before: $before
      first: $first
      last: $last
    ) {
      edges {
        node {
          id
          userId
          isEphemeral
          createdAt
          updatedAt
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

export const listMyThreads = graphql(/* GraphQL */ `
  query ListMyThreads(
    $after: String
    $before: String
    $first: Int
    $last: Int
  ) {
    myThreads(after: $after, before: $before, first: $first, last: $last) {
      edges {
        node {
          id
          userId
          isEphemeral
          createdAt
          updatedAt
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

export const listThreadMessages = graphql(/* GraphQL */ `
  query ListThreadMessages(
    $threadId: ID!
    $after: String
    $before: String
    $first: Int
    $last: Int
  ) {
    threadMessages(
      threadId: $threadId
      after: $after
      before: $before
      first: $first
      last: $last
    ) {
      edges {
        node {
          id
          threadId
          codeSourceId
          role
          content
          attachment {
            code {
              gitUrl
              commit
              filepath
              language
              content
              startLine
            }
            clientCode {
              filepath
              content
              startLine
            }
            doc {
              __typename
              ... on MessageAttachmentWebDoc {
                title
                link
                content
              }
              ... on MessageAttachmentIssueDoc {
                title
                link
                author {
                  id
                  email
                  name
                }
                body
                closed
              }
              ... on MessageAttachmentPullDoc {
                title
                link
                author {
                  id
                  email
                  name
                }
                body
                merged
              }
              ... on MessageAttachmentCommitDoc {
                sha
                message
                author {
                  id
                  email
                  name
                }
                authorAt
              }
              ... on MessageAttachmentPageDoc {
                title
                link
                content
              }
              ... on MessageAttachmentIngestedDoc {
                id
                title
                body
                ingestedDocLink: link
              }
            }
            codeFileList {
              fileList
              truncated
            }
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

export const setThreadPersistedMutation = graphql(/* GraphQL */ `
  mutation SetThreadPersisted($threadId: ID!) {
    setThreadPersisted(threadId: $threadId)
  }
`)

export const deleteThreadMutation = graphql(/* GraphQL */ `
  mutation DeleteThread($id: ID!) {
    deleteThread(id: $id)
  }
`)

export const notificationsQuery = graphql(/* GraphQL */ `
  query Notifications {
    notifications {
      id
      content
      read
      createdAt
    }
  }
`)

export const ldapCredentialQuery = graphql(/* GraphQL */ `
  query LdapCredential {
    ldapCredential {
      host
      port
      bindDn
      baseDn
      userFilter
      encryption
      skipTlsVerify
      emailAttribute
      nameAttribute
      createdAt
      updatedAt
    }
  }
`)

export const oauthCredential = graphql(/* GraphQL */ `
  query OAuthCredential($provider: OAuthProvider!) {
    oauthCredential(provider: $provider) {
      provider
      clientId
      createdAt
      updatedAt
    }
  }
`)

export const repositorySourceListQuery = graphql(/* GraphQL */ `
  query RepositorySourceList {
    repositoryList {
      id
      name
      kind
      gitUrl
      sourceId
      sourceName
      sourceKind
    }
  }
`)

export const listPages = graphql(/* GraphQL */ `
  query ListPages(
    $ids: [ID!]
    $after: String
    $before: String
    $first: Int
    $last: Int
  ) {
    pages(
      ids: $ids
      after: $after
      before: $before
      first: $first
      last: $last
    ) {
      edges {
        node {
          id
          authorId
          title
          codeSourceId
          content
          createdAt
          updatedAt
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

export const listPageSections = graphql(/* GraphQL */ `
  query ListPageSections(
    $pageId: ID!
    $after: String
    $before: String
    $first: Int
    $last: Int
  ) {
    pageSections(
      pageId: $pageId
      after: $after
      before: $before
      first: $first
      last: $last
    ) {
      edges {
        node {
          id
          pageId
          title
          content
          position
          attachments {
            code {
              __typename
              gitUrl
              commit
              filepath
              language
              content
              startLine
            }
            codeFileList {
              __typename
              fileList
              truncated
            }
            doc {
              __typename
              ... on AttachmentWebDoc {
                title
                link
                content
              }
              ... on AttachmentIssueDoc {
                title
                link
                author {
                  id
                  email
                  name
                }
                body
                closed
              }
              ... on AttachmentPullDoc {
                title
                link
                author {
                  id
                  email
                  name
                }
                body
                merged
              }
              ... on AttachmentCommitDoc {
                sha
                message
                author {
                  id
                  email
                  name
                }
                authorAt
              }
              ... on AttachmentPageDoc {
                link
                title
                content
              }
              ... on AttachmentIngestedDoc {
                id
                title
                body
                ingestedDocLink: link
              }
            }
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

export const brandingSettingQuery = graphql(/* GraphQL */ `
  query GeneralBrandingQuery {
    brandingSetting {
      brandingLogo
      brandingIcon
    }
  }
`)
