import { Message } from 'ai'

import type { Chat } from '@/lib/types'
import { nanoid } from '@/lib/utils'

import { useChatStore } from './chat-store'

const get = useChatStore.getState
const set = useChatStore.setState

export const updateHybrated = (state: boolean) => {
  set(() => ({ _hasHydrated: state }))
}
export const setActiveChatId = (id: string) => {
  set(() => ({ activeChatId: id }))
}

export const addChat = (_id?: string, title?: string) => {
  const id = _id ?? nanoid()
  set(state => ({
    activeChatId: id,
    chats: [
      {
        id,
        title: title ?? '',
        messages: [],
        createdAt: new Date(),
        userId: '',
        path: ''
      },
      ...(state.chats || [])
    ]
  }))
}

export const deleteChat = (id: string) => {
  set(state => {
    return {
      activeChatId: nanoid(),
      chats: state.chats?.filter(chat => chat.id !== id)
    }
  })
}

export const clearChats = () => {
  set(() => ({
    activeChatId: nanoid(),
    chats: []
  }))
}

export const updateMessages = (id: string, messages: Message[]) => {
  set(state => ({
    chats: state.chats?.map(chat => {
      if (chat.id === id) {
        return {
          ...chat,
          messages
        }
      }
      return chat
    })
  }))
}

export const updateChat = (id: string, chat: Partial<Chat>) => {
  set(state => ({
    chats: state.chats?.map(c => {
      if (c.id === id) {
        return {
          ...c,
          ...chat
        }
      }
      return c
    })
  }))
}
