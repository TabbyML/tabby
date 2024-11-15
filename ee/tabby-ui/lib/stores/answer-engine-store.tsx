import { create } from 'zustand'

export interface AnswerEngineState {
  threadsPageNo: number
}

const initialState: AnswerEngineState = {
  threadsPageNo: 1
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
