'use client'

import Link from 'next/link'
import { useParams } from 'next/navigation'

import { RepositoryKind } from '@/lib/gql/generates/graphql'
import { IconFolderGit, IconGitHub, IconGitLab } from '@/components/ui/icons'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'

export default function GitTabsHeader() {
  const params = useParams<{ kind?: RepositoryKind }>()
  const defaultValue = params?.kind ? params.kind?.toLowerCase() : 'git'

  return (
    <Tabs value={defaultValue}>
      <div className="sticky top-0 mb-4 flex">
        <TabsList className="grid grid-cols-3">
          <TabsTrigger value="git" asChild>
            <Link href="/settings/repository/git">
              <span className="ml-2">Git</span>
            </Link>
          </TabsTrigger>
          <TabsTrigger value="github" asChild>
            <Link href="/settings/repository/github">
              <IconGitHub />
              <span className="ml-2">GitHub</span>
            </Link>
          </TabsTrigger>
          <TabsTrigger value="gitlab" asChild>
            <Link href="/settings/repository/gitlab">
              <IconGitLab />
              <span className="ml-2">GitLab</span>
            </Link>
          </TabsTrigger>
        </TabsList>
      </div>
    </Tabs>
  )
}
