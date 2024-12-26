import { useChatStore } from './chat-store'

const set = useChatStore.setState

export const setActiveChatId = (id: string) => {
  set(() => ({ activeChatId: id }))
}

export const updateSelectedModel = (model: string | undefined) => {
  set(() => ({ selectedModel: model }))
}

export const updateSelectedRepo = (sourceId: string | undefined) => {
  set(() => ({ selectedRepo: sourceId }))
}

export const updateEnableActiveSelection = (enable: boolean) => {
  set(() => ({ enableActiveSelection: enable }))
}
