'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'

import { IconGitHub } from '@/components/ui/icons'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'

export default function GitTabsHeader() {
  const pathname = usePathname()
  const defualtValue = pathname.indexOf('github') >= 0 ? 'github' : 'git'

  return (
    <Tabs defaultValue={defualtValue}>
      <div className="sticky top-0 mb-4 flex">
        <TabsList className="grid grid-cols-2">
          <TabsTrigger value="git" asChild>
            <Link href="/settings/repository/git">Git</Link>
          </TabsTrigger>
          <TabsTrigger value="github" asChild>
            <Link href="/settings/repository/github">
              <IconGitHub />
              <span className="ml-2">GitHub</span>
            </Link>
          </TabsTrigger>
        </TabsList>
      </div>
    </Tabs>
  )
}
