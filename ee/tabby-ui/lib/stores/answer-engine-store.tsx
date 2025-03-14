import { create } from 'zustand'

export interface AnswerEngineState {
  threadsPageNo: number
  myThreadsPageNo: number
  threadsTab: 'all' | 'mine'
}

const initialState: AnswerEngineState = {
  threadsPageNo: 1,
  myThreadsPageNo: 1,
  threadsTab: 'all'
}

export const useAnswerEngineStore = create<AnswerEngineState>()(() => {
  return { ...initialState }
})

const set = useAnswerEngineStore.setState

export const resetThreadsPageNo = () => {
  return set(() => ({ threadsPageNo: initialState.threadsPageNo }))
}

export const setThreadsPageNo = (pageNo: number) => {
  return set(() => ({ threadsPageNo: pageNo }))
}

export const setMyThreadsPageNo = (pageNo: number) => {
  return set(() => ({ myThreadsPageNo: pageNo }))
}

export const resetMyThreadsPageNo = () => {
  return set(() => ({ myThreadsPageNo: initialState.threadsPageNo }))
}

export const setThreadsTab = (tab: 'all' | 'mine') => {
  return set(() => ({ threadsTab: tab }))
}
