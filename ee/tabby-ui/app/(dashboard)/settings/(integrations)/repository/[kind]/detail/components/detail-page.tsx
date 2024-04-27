'use client'

import React from 'react'

import { RepositoryKind } from '@/lib/gql/generates/graphql'

import { useRepositoryKind } from '../../hooks/use-repository-kind'
import GithubProviderDetail from './github-provider-detail'
import GitlabProviderDetail from './gitlab-provider-detail'

const DetailPage: React.FC = () => {
  const kind = useRepositoryKind()

  if (kind === RepositoryKind.Github) {
    return <GithubProviderDetail />
  }

  if (kind === RepositoryKind.Gitlab) {
    return <GitlabProviderDetail />
  }

  return <div>404</div>
}

export default DetailPage
