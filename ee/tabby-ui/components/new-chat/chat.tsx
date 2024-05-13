import React from 'react'
import { useChat, type Message, type UseChatHelpers } from 'ai/react'
import { toast } from 'sonner'
import { nanoid } from 'ai'
import { cn } from '@/lib/utils'
import { ChatScrollAnchor } from '../chat-scroll-anchor'
import { EmptyScreen } from '../empty-screen'
import { ChatList } from './chat-list'
import { find, findIndex } from 'lodash-es'
import { ChatPanel } from '../chat-panel'


interface LineRange {
  start: number
  end: number
}

export interface FileContext {
  kind: "file"
  range: LineRange,
  filename: string,
  link: string
  providerId: string
  repositoryId: string
  // FIXME(jueliang): add code snippet here for client side mock
  content: string
}

export type ChatContext = FileContext

export interface UserMessage {
  id: string
  message: string
  selectContext?: ChatContext
  relevantContext?: Array<ChatContext>
}

export interface AssistantMessage {
  id: string
  message: string
}

export interface QuestionAnswerPair {
  user: UserMessage,
  assistant: AssistantMessage
}

function QuestionAnswerItem({ message }: { message: QuestionAnswerPair }) {
  // todo refenrence context ?
  return (
    <div>QuestionAnswerItem</div>
  )
}

function UserMessageCard({ message }: { message: UserMessage }) {
  return <div>user message card</div>
}

function AssistantMessageCard({ message }: { message: AssistantMessage }) {
  return <div>assistant message card</div>
}

function toMessages(qaPairs: QuestionAnswerPair[] | undefined): Message[] {
  if (!qaPairs?.length) return []
  let result: Message[] = []
  for (let pair of qaPairs) {
    let { user, assistant } = pair
    if (user) {
      result.push(userMessageToMessage(user))
    }
    if (assistant) {
      result.push({
        role: 'assistant',
        id: assistant.id,
        content: assistant.message
      })
    }
  }
  return result
}

function userMessageToMessage(userMessage: UserMessage): Message {
  // todo deal with undefined
  const { selectContext, relevantContext, message, id } = userMessage
  return {
    id,
    role: 'user',
    content: message + fileContextToMessageContent(selectContext)
  }
}

function fileContextToMessageContent(context: FileContext | undefined): string {
  if (!context) return ''
  const { link, range, filename, content, providerId, repositoryId } = context
  const metaMessage = `is_reference=1 file_name=${filename} provider_id=${providerId} repository_id=${repositoryId} start_line=${range.start} end_line=${range.end} file_name=${filename} link=${link}`
  return `\n${'```'} ${metaMessage}\n${content}\n${'```'}\n`
}

// function toQAPaires(messages: Message[] | undefined): QuestionAnswerPair[] {
//   if (!messages?.length) return []
//   let lastUserMessage: Message | undefined
//   let lastAssisantMessage: Message | undefined

//   let result: QuestionAnswerPair[] = []
//   let len = messages.length
//   for (let message of messages) {
//     if (message.role === 'user') {
//       if (!lastUserMessage) {
//         lastUserMessage = message
//       } else {
//         lastUserMessage = 
//       }
//     } else {

//     }
//   }
// }

// function mergeUserMessage(message1: Message, message2: Message): UserMessage {
//   return {
//     message: `${message1.content}\n${message2.content}`,
//   }
// }

// function messageToUserMessage(message: Message): UserMessage {
//   return {
//     message: message.content,
//     relevantContext

//   }
// }

function getSelectContextFromMessage() {

}

export interface ChatRef extends UseChatHelpers { }

interface ChatProps extends React.ComponentProps<'div'> {
  initialMessages?: QuestionAnswerPair[]
  id?: string
  // todo data type
  onThreadUpdates: (id: string, messages: Message[]) => void
}


function Chat({ id, initialMessages, onThreadUpdates, className }: ChatProps) {

  const [isStreamResponsePending, setIsStreamResponsePending] =
    React.useState(false)
  const transformedMessages = toMessages(initialMessages)

  const useChatHelpers = useChat({
    initialMessages: transformedMessages,
    id,
    body: {
      id
    },
    onResponse(response) {
      if (response.status === 401) {
        toast.error(response.statusText)
      }
    }
  })

  const {
    messages,
    append,
    reload,
    stop,
    isLoading,
    input,
    setInput,
    setMessages
  } = useChatHelpers

  const handleSubmit = async (value: string) => {
    // if (findIndex(chats, { id }) === -1) {
    //   addChat(id, truncateText(value))
    // } else if (selectedMessageId) {
    //   let messageIdx = findIndex(messages, { id: selectedMessageId })
    //   setMessages(messages.slice(0, messageIdx))
    //   setSelectedMessageId(undefined)
    // }
    await append({
      id: nanoid(),
      content: value,
      role: 'user'
    })
    // todo onthreadUpdate
  }
  const onEditMessage = (messageId: string) => {
    const message = find(messages, { id: messageId })
    if (message) {
      setInput(message.content)
      // setSelectedMessageId(messageId)
    }
  }

  const onDeleteMessage = (messageId: string) => {
    const message = find(messages, { id: messageId })
    if (message) {
      setMessages(messages.filter(m => m.id !== messageId))
    }
  }

  const onRegenerateResponse = (messageId: string) => {
    const messageIndex = findIndex(messages, { id: messageId })
    const prevMessage = messages?.[messageIndex - 1]
    if (prevMessage?.role === 'user') {
      setMessages(messages.slice(0, messageIndex - 1))
      append(prevMessage)
    }
  }

  const onStop = () => {
    setIsStreamResponsePending(false)
    stop()
  }


  const handleMessageAction = (
    messageId: string,
    actionType: 'delete' | 'regenerate'
  ) => {
    switch (actionType) {
      case 'delete':
        onDeleteMessage(messageId)
        break
      case 'regenerate':
        onRegenerateResponse(messageId)
        break
      default:
        break
    }
  }

  React.useEffect(() => {
    if (id) {
      onThreadUpdates?.(id, messages)
    }
  }, [messages])

  return (
    <div className="flex justify-center overflow-x-hidden">
      <div className="w-full max-w-2xl px-4">
        <div className={cn('pb-[200px] pt-4 md:pt-10', className)}>
          {messages.length ? (
            <>
              <ChatList
                messages={messages}
                handleMessageAction={handleMessageAction}
                isStreamResponsePending={isStreamResponsePending}
              />
              <ChatScrollAnchor trackVisibility={isLoading} />
            </>
          ) : (
            <EmptyScreen setInput={setInput} />
          )}
        </div>
        <ChatPanel
          onSubmit={handleSubmit}
          className="fixed inset-x-0 bottom-0 lg:ml-[280px]"
          id={id}
          isLoading={isLoading}
          stop={onStop}
          append={append}
          reload={reload}
          messages={messages}
          input={input}
          setInput={setInput}
          setMessages={setMessages}
        />
      </div>
    </div>
  )
}