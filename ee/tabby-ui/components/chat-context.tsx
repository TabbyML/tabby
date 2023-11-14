'use client'

import React from 'react'
import { Message, UseChatHelpers, useChat } from 'ai/react'
import { toast } from 'react-hot-toast'

export interface ChatContextValue extends UseChatHelpers {
  id: string | undefined
}

export const ChatContext = React.createContext({} as ChatContextValue)
export interface ChatContextProviderProps {
  id: string | undefined
  initialMessages?: Message[]
}

export const ChatContextProvider: React.FC<
  React.PropsWithChildren<ChatContextProviderProps>
> = ({ children, id, initialMessages }) => {
  const chatHelpers = useChat({
    initialMessages,
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

  return (
    <ChatContext.Provider value={{ ...chatHelpers, id }}>
      {children}
    </ChatContext.Provider>
  )
}
