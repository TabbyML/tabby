'use client'

import React from 'react'
import Link from 'next/link'
import { useParams } from 'next/navigation'

import { IntegrationKind, RepositoryKind } from '@/lib/gql/generates/graphql'
import { IconGitHub, IconGitLab } from '@/components/ui/icons'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'

import { PROVIDER_KIND_METAS } from '../constants'

export default function GitTabsHeader() {
  const params = useParams<{ kind?: RepositoryKind }>()
  const defaultValue = React.useMemo(() => {
    return params?.kind ? params.kind?.toLowerCase() : 'git'
  }, [params?.kind])

  return (
    <Tabs value={defaultValue}>
      <div className="sticky top-0 mb-4 flex">
        <TabsList className="grid grid-cols-5">
          <TabsTrigger value="git" asChild>
            <Link href="/settings/providers/git">
              <span className="ml-2">Git</span>
            </Link>
          </TabsTrigger>
          {PROVIDER_KIND_METAS.map(provider => {
            return (
              <TabsTrigger value={provider.name} asChild key={provider.name}>
                <Link href={`/settings/providers/${provider.name}`}>
                  <ProviderIcon kind={provider.enum} />
                  <span className="ml-2">{provider.meta.displayName}</span>
                </Link>
              </TabsTrigger>
            )
          })}
        </TabsList>
      </div>
    </Tabs>
  )
}

function ProviderIcon({ kind }: { kind: IntegrationKind }) {
  switch (kind) {
    case IntegrationKind.Github:
    case IntegrationKind.GithubSelfHosted:
      return <IconGitHub />
    case IntegrationKind.Gitlab:
    case IntegrationKind.GitlabSelfHosted:
      return <IconGitLab />
    default:
      return null
  }
}
