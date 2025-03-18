import { graphql } from '@/lib/gql/generates'

export const createThreadToPageRunSubscription = graphql(/* GraphQL */ `
  subscription createThreadToPageRun($threadId: ID!) {
    createThreadToPageRun(threadId: $threadId) {
      __typename
      ... on PageCreated {
        id
        authorId
        title
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
      }
      ... on PageSectionContentDelta {
        id
        delta
      }
      ... on PageSectionContentCompleted {
        id
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
      ... on PageSection {
        id
        title
        position
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
      }
      ... on PageSectionContentDelta {
        id
        delta
      }
      ... on PageSectionContentCompleted {
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
      }
      ... on PageSectionContentDelta {
        id
        delta
      }
      ... on PageSectionContentCompleted {
        id
      }
      ... on PageCompleted {
        id
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
