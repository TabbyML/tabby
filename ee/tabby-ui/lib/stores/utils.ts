import { Chat } from '@/lib/types'

export const getChatById = (
  chats: Chat[] | undefined,
  chatId: string | undefined
): Chat | undefined => {
  if (!Array.isArray(chats) || !chatId) return undefined
  return chats.find(c => c.id === chatId)
}
