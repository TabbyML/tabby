import {
  AttachmentCode,
  AttachmentCodeFileList,
  ListPageSectionsQuery,
  ListPagesQuery,
  Maybe,
  MessageCodeSearchHit
} from '@/lib/gql/generates/graphql'

export type PageItem = ListPagesQuery['pages']['edges'][0]['node']
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
  }
}
