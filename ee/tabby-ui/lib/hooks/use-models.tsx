'use client'

import { useEffect } from 'react'
import { Maybe } from 'graphql/jsutils/Maybe'
import useSWR, { SWRResponse } from 'swr'
import { useStore } from 'zustand'

import fetcher from '@/lib/tabby/fetcher'

import { updateSelectedModel, useChatStore } from '../stores/chat-store'

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
  const { data: modelData, isLoading } = useModels()

  const selectedModel = useStore(useChatStore, state => state.selectedModel)

  useEffect(() => {
    if (!isLoading) {
      // init model
      const validModel = getModelFromModelInfo(selectedModel, modelData?.chat)
      updateSelectedModel(validModel)
    }
  }, [isLoading])

  return {
    // fetching model data or trying to get selected model from localstorage
    isFetchingModels: isLoading,
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
