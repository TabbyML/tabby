import {
  Maybe,
  Message,
  MessageAttachmentClientCode,
  MessageAttachmentCodeFileList,
  ThreadAssistantMessageReadingCode
} from '@/lib/gql/generates/graphql'
import { AttachmentCodeItem, AttachmentDocItem } from '@/lib/types'

export type ConversationMessage = Omit<
  Message,
  '__typename' | 'updatedAt' | 'createdAt' | 'attachment' | 'threadId'
> & {
  threadId?: string
  threadRelevantQuestions?: Maybe<string[]>
  error?: string
  attachment?: {
    clientCode?: Maybe<Array<MessageAttachmentClientCode>> | undefined
    code: Maybe<Array<AttachmentCodeItem>> | undefined
    doc: Maybe<Array<AttachmentDocItem>> | undefined
    codeFileList?: Maybe<MessageAttachmentCodeFileList>
  }
  readingCode?: ThreadAssistantMessageReadingCode
  isReadingCode?: boolean
  isReadingFileList?: boolean
  isReadingDocs?: boolean
}
export type ConversationPair = {
  question: ConversationMessage | null
  answer: ConversationMessage | null
}
