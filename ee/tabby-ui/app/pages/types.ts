import {
  AttachmentCode,
  AttachmentCodeFileList,
  ChatCompletionMessage,
  ListPageSectionsQuery,
  ListPagesQuery,
  Maybe,
  MessageCodeSearchHit
} from '@/lib/gql/generates/graphql'
import { AttachmentDocItem } from '@/lib/types'

export type PageItem = ListPagesQuery['pages']['edges'][0]['node']

export type SectionDebugDataItem = {
  attachmentCodeQuery?: {
    sourceId: string
    query: string
  }
  attachmentDocQuery?: {
    sourceIds: string[]
    query: string
  }
  generateSectionContentMessages?: ChatCompletionMessage[]
}

export type DebugData = {
  generatePageTitleMessages?: ChatCompletionMessage[]
  generatePageContentMessages?: ChatCompletionMessage[]
  generateSectionTitlesMessages?: ChatCompletionMessage[]
  sections?: Array<
    {
      id: string
    } & SectionDebugDataItem
  >
}

export type SectionItem = Omit<
  ListPageSectionsQuery['pageSections']['edges'][0]['node'],
  '__typename' | 'attachments'
> & {
  attachments: {
    __typename?: 'SectionAttachment'
    code: Array<
      AttachmentCode & {
        scores?: MessageCodeSearchHit['scores']
      }
    >
    codeFileList?: Maybe<AttachmentCodeFileList>
    doc?: Maybe<AttachmentDocItem[]>
  }
}
