import { create } from 'zustand'

export interface PageState {
  pendingThreadId: string | undefined
  pendingThreadTitle: string | undefined
}

const initialState: PageState = {
  pendingThreadId: undefined,
  pendingThreadTitle: undefined
}

export const usePageStore = create<PageState>()(() => ({ ...initialState }))

const set = usePageStore.setState

export const updatePendingThreadId = (threadId: string | undefined) => {
  set(() => ({ pendingThreadId: threadId }))
}

export const updatePendingThreadTitle = (title: string | undefined) => {
  set(() => ({ pendingThreadTitle: title }))
}

export const updatePendingThread = ({
  threadId,
  title
}: {
  threadId: string
  title: string
}) => {
  set(() => ({
    pendingThreadId: threadId,
    pendingThreadTitle: title
  }))
}

export const clearPendingThread = () => {
  set(() => ({
    pendingThreadId: undefined,
    pendingThreadTitle: undefined
  }))
}
