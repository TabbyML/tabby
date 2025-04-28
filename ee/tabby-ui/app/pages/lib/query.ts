import { graphql } from '@/lib/gql/generates'

export const createThreadToPageRunSubscription = graphql(/* GraphQL */ `
  subscription createThreadToPageRun($input: CreateThreadToPageRunInput!) {
    createThreadToPageRun(input: $input) {
      __typename
      ... on PageCreated {
        id
        authorId
        title
        debugData {
          __typename
          generatePageTitleMessages {
            role
            content
          }
        }
      }
      ... on PageContentDelta {
        delta
      }
      ... on PageContentCompleted {
        id
      }
      ... on PageSectionsCreated {
        sections {
          id
          position
          title
        }
        debugData {
          __typename
          generateSectionTitlesMessages {
            role
            content
          }
        }
      }
      ... on PageSectionAttachmentCodeFileList {
        id
        codeFileList {
          __typename
          fileList
          truncated
        }
      }
      ... on PageSectionAttachmentCode {
        id
        codes {
          code {
            __typename
            gitUrl
            commit
            filepath
            language
            content
            startLine
          }
          scores {
            rrf
            bm25
            embedding
          }
        }
        debugData {
          sourceId
          query
        }
      }
      ... on PageSectionAttachmentDoc {
        id
        doc {
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
          score
        }
        debugData {
          sourceIds
          query
        }
      }
      ... on PageSectionContentDelta {
        id
        delta
      }
      ... on PageSectionContentCompleted {
        id
        debugData {
          __typename
          generateSectionContentMessages {
            role
            content
          }
        }
      }
      ... on PageCompleted {
        id
      }
    }
  }
`)

export const createPageRunSubscription = graphql(/* GraphQL */ `
  subscription createPageRun($input: CreatePageRunInput!) {
    createPageRun(input: $input) {
      __typename
      ... on PageCreated {
        id
        authorId
        title
        debugData {
          __typename
          generatePageTitleMessages {
            role
            content
          }
        }
      }
      ... on PageContentDelta {
        delta
      }
      ... on PageContentCompleted {
        id
        debugData {
          __typename
          generatePageContentMessages {
            role
            content
          }
        }
      }
      ... on PageSectionsCreated {
        sections {
          id
          position
          title
        }
        debugData {
          __typename
          generateSectionTitlesMessages {
            role
            content
          }
        }
      }
      ... on PageSectionAttachmentCodeFileList {
        id
        codeFileList {
          __typename
          fileList
          truncated
        }
      }
      ... on PageSectionAttachmentCode {
        id
        codes {
          code {
            __typename
            gitUrl
            commit
            filepath
            language
            content
            startLine
          }
          scores {
            rrf
            bm25
            embedding
          }
        }
        debugData {
          sourceId
          query
        }
      }
      ... on PageSectionAttachmentDoc {
        id
        doc {
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
          score
        }
        debugData {
          sourceIds
          query
        }
      }
      ... on PageSectionContentDelta {
        id
        delta
      }
      ... on PageSectionContentCompleted {
        id
        debugData {
          __typename
          generateSectionContentMessages {
            role
            content
          }
        }
      }
      ... on PageCompleted {
        id
      }
    }
  }
`)

export const createPageSectionRunSubscription = graphql(/* GraphQL */ `
  subscription createPageSectionRun($input: CreatePageSectionRunInput!) {
    createPageSectionRun(input: $input) {
      __typename
      ... on PageSectionCreated {
        id
        title
        position
        debugData {
          __typename
          generateSectionTitlesMessages {
            role
            content
          }
        }
      }
      ... on PageSectionAttachmentCodeFileList {
        id
        codeFileList {
          fileList
          truncated
        }
      }
      ... on PageSectionAttachmentCode {
        id
        codes {
          code {
            __typename
            gitUrl
            commit
            filepath
            language
            content
            startLine
          }
          scores {
            rrf
            bm25
            embedding
          }
        }
        debugData {
          sourceId
          query
        }
      }
      ... on PageSectionAttachmentDoc {
        id
        doc {
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
          score
        }
        debugData {
          sourceIds
          query
        }
      }
      ... on PageSectionContentDelta {
        id
        delta
      }
      ... on PageSectionContentCompleted {
        id
        debugData {
          __typename
          generateSectionContentMessages {
            role
            content
          }
        }
      }
    }
  }
`)

export const deletePageSectionMutation = graphql(/* GraphQL */ `
  mutation DeletePageSection($sectionId: ID!) {
    deletePageSection(sectionId: $sectionId)
  }
`)

export const movePageSectionPositionMutation = graphql(/* GraphQL */ `
  mutation movePageSection($id: ID!, $direction: MoveSectionDirection!) {
    movePageSection(id: $id, direction: $direction)
  }
`)
