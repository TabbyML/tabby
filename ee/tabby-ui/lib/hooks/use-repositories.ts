'use client'

import { useMemo } from 'react'
import { useQuery } from 'urql'

import {
  updateSelectedRepoSourceId,
  useChatStore
} from '@/lib/stores/chat-store'
import { repositorySourceListQuery } from '@/lib/tabby/query'

export function useRepositorySources() {
  return useQuery({
    query: repositorySourceListQuery
  })
}

export function useSelectedRepository() {
  const [{ data, fetching }] = useRepositorySources()
  const repos = data?.repositoryList
  const repoId = useChatStore(state => state.selectedRepoSourceId)

  const onSelectRepository = (sourceId: string | undefined) => {
    updateSelectedRepoSourceId(sourceId)
  }

  const selectedRepository = useMemo(() => {
    if (!repos?.length || !repoId) return undefined

    return repos.find(repo => repo.sourceId === repoId)
  }, [repos, repoId])

  return {
    repos,
    isFetchingRepositories: fetching,
    selectedRepository,
    onSelectRepository
  }
}
