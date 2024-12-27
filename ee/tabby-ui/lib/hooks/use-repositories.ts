'use client'

import { useMemo } from 'react'
import { useQuery } from 'urql'

import { graphql } from '@/lib/gql/generates'

import { useChatStore } from '../stores/chat-store'

const repositorySourceListQuery = graphql(/* GraphQL */ `
  query RepositorySourceList {
    repositoryList {
      id
      name
      kind
      gitUrl
      sourceId
      sourceName
      sourceKind
    }
  }
`)

export function useRepositorySources() {
  return useQuery({
    query: repositorySourceListQuery
  })
}

export function useSelectedRepository() {
  const [{ data, fetching }] = useRepositorySources()
  const repos = data?.repositoryList
  const repoId = useChatStore(state => state.selectedCodeSourceId)

  const selectedRepository = useMemo(() => {
    if (!repos?.length || !repoId) return undefined

    return repos.find(repo => repo.sourceId === repoId)
  }, [repos, repoId])

  return {
    repos,
    isFetchingRepositories: fetching,
    selectedRepository
  }
}
