'use client'

import React, { useState } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'

import { IntegrationKind } from '@/lib/gql/generates/graphql'
import { cn } from '@/lib/utils'
import { IconFileText, IconGitHub, IconGitLab } from '@/components/ui/icons'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'

import { PROVIDER_KIND_METAS } from '../constants'

export default function GitTabsHeader() {
  const pathname = usePathname()
  const defaultValue = React.useMemo(() => {
    const matcher = pathname.match(/^\/settings\/providers\/([\w-]+)/)?.[1]
    return matcher?.toLowerCase() ?? 'git'
  }, [pathname])
  const [tab, setTab] = useState(defaultValue || '')

  return (
    <Tabs value={defaultValue} onValueChange={setTab}>
      <div className="sticky top-0 mb-4 flex">
        <TabsList className="grid h-20 grid-cols-3 lg:h-10 lg:grid-cols-6">
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
          <TabsTrigger value="web" asChild>
            <Link
              href="/settings/providers/web"
              className="relative overflow-hidden"
            >
              <IconFileText />
              <span className="ml-2">Web</span>
              <span
                className={cn(
                  'absolute -right-8 top-1 mr-3 rotate-45 rounded-none border-none bg-muted py-0.5 pl-6 pr-5 text-xs text-muted-foreground',
                  {
                    'opacity-100': tab === 'web',
                    'opacity-0': tab !== 'web'
                  }
                )}
                style={{ transition: 'opacity 0.35s ease-out 0.15s' }}
              >
                Beta
              </span>
            </Link>
          </TabsTrigger>
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
