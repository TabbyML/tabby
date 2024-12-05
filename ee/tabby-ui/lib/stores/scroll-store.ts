import { create } from 'zustand'

export interface ScrollState {
  homePage: number | undefined
}

const initialState: ScrollState = {
  homePage: undefined
}

export const useScrollStore = create<ScrollState>()(() => {
  return { ...initialState }
})

const set = useScrollStore.setState

export const setHomeScrollPosition = (scrollTop: number) => {
  return set(() => ({ homePage: scrollTop }))
}

export const clearHomeScrollPosition = () => {
  return set(() => ({ homePage: undefined }))
}
