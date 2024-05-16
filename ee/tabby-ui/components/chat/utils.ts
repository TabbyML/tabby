import { Message } from 'ai'

export function mergeMessagesByRole(body: BodyInit | null | undefined) {
  if (typeof body !== 'string') return body
  try {
    const bodyObject = JSON.parse(body)
    let messages: Message[] = bodyObject.messages?.slice()
    if (Array.isArray(messages) && messages.length > 1) {
      let previewCursor = 0
      let curCursor = 1
      while (curCursor < messages.length) {
        let prevMessage = messages[previewCursor]
        let curMessage = messages[curCursor]
        if (curMessage.role === prevMessage.role) {
          messages = [
            ...messages.slice(0, previewCursor),
            {
              ...prevMessage,
              content: [prevMessage.content, curMessage.content].join('\n')
            },
            ...messages.slice(curCursor + 1)
          ]
        } else {
          previewCursor = curCursor++
        }
      }
      return JSON.stringify({
        ...bodyObject,
        messages
      })
    } else {
      return body
    }
  } catch (e) {
    return body
  }
}
