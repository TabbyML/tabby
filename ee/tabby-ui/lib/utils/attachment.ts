import { AttachmentDocItem } from '../types'

export function isAttachmentWebDoc(attachment: AttachmentDocItem) {
  return (
    attachment.__typename === 'AttachmentWebDoc' ||
    attachment.__typename === 'MessageAttachmentWebDoc'
  )
}

export function isAttachmentCommitDoc(attachment: AttachmentDocItem) {
  return (
    attachment.__typename === 'AttachmentCommitDoc' ||
    attachment.__typename === 'MessageAttachmentCommitDoc'
  )
}

export function isAttachmentPullDoc(attachment: AttachmentDocItem) {
  return (
    attachment.__typename === 'AttachmentPullDoc' ||
    attachment.__typename === 'MessageAttachmentPullDoc'
  )
}

export function isAttachmentIssueDoc(attachment: AttachmentDocItem) {
  return (
    attachment.__typename === 'AttachmentIssueDoc' ||
    attachment.__typename === 'MessageAttachmentIssueDoc'
  )
}

export function isAttachmentPageDoc(attachment: AttachmentDocItem) {
  return (
    attachment.__typename === 'AttachmentPageDoc' ||
    attachment.__typename === 'MessageAttachmentPageDoc'
  )
}

export function isAttachmentIngestedDoc(attachment: AttachmentDocItem) {
  return (
    attachment.__typename === 'AttachmentIngestedDoc' ||
    attachment.__typename === 'MessageAttachmentIngestedDoc'
  )
}

export function getAttachmentDocContent(attachment: AttachmentDocItem) {
  switch (attachment.__typename) {
    case 'MessageAttachmentWebDoc':
    case 'AttachmentWebDoc':
    case 'AttachmentPageDoc':
    case 'MessageAttachmentPageDoc':
      return attachment.content
    case 'MessageAttachmentIssueDoc':
    case 'MessageAttachmentPullDoc':
    case 'AttachmentIssueDoc':
    case 'AttachmentPullDoc':
    case 'AttachmentIngestedDoc':
    case 'MessageAttachmentIngestedDoc':
      return attachment.body
    case 'MessageAttachmentCommitDoc':
    case 'AttachmentCommitDoc':
      return attachment.message
    default:
      return ''
  }
}
