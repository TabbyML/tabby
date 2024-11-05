'use client'

import { useEffect } from 'react'
import { Maybe } from 'graphql/jsutils/Maybe'
import useSWR, { SWRResponse } from 'swr'

import fetcher from '@/lib/tabby/fetcher'

import { updateSelectedModel } from '../stores/chat-actions'
import { useChatStore } from '../stores/chat-store'
import { useStore } from './use-store'

export interface ModelInfo {
  completion: Maybe<Array<string>>
  chat: Maybe<Array<string>>
}

export function useModels(): SWRResponse<ModelInfo> {
  return useSWR(
    '/v1beta/models',
    (url: string) => {
      return fetcher(url, {
        errorHandler: () => {
          throw new Error('Fetch supported model failed.')
        }
      })
    },
    {
      shouldRetryOnError: false
    }
  )
}

export function useSelectedModel() {
  const { data: modelData, isLoading: isFetchingModel } = useModels()
  const isModelHydrated = useStore(useChatStore, state => state._hasHydrated)

  const selectedModel = useStore(useChatStore, state => state.selectedModel)

  // once model hydrated, try to init model
  useEffect(() => {
    if (isModelHydrated && !isFetchingModel) {
      // check if current model is valid
      const validModel = getModelFromModelInfo(selectedModel, modelData?.chat)
      if (selectedModel !== validModel) {
        updateSelectedModel(validModel)
      }
    }
  }, [isModelHydrated, isFetchingModel])

  return {
    // fetching model data or trying to get selected model from localstorage
    isModelLoading: isFetchingModel || !isModelHydrated,
    selectedModel,
    models: modelData?.chat
  }
}

export function getModelFromModelInfo(
  model: string | undefined,
  models: Maybe<Array<string>> | undefined
) {
  if (!models?.length) return undefined

  const isValidModel = !!model && models.includes(model)
  if (isValidModel) {
    return model
  }

  // return the first model by default
  return models[0]
}
