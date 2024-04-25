'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'

import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'

export default function GitTabsHeader() {
  const pathname = usePathname()
  const defualtValue = pathname.startsWith('/settings/git/generic')
    ? 'generic'
    : 'gitops'

  return (
    <Tabs defaultValue={defualtValue}>
      <div className="sticky top-0 mb-4 flex">
        <TabsList className="grid grid-cols-2">
          <TabsTrigger value="gitops" asChild>
            <Link href="/settings/git/gitops">Gitops</Link>
          </TabsTrigger>
          <TabsTrigger value="generic" asChild>
            <Link href="/settings/git/generic">Generic Git Repositories</Link>
          </TabsTrigger>
        </TabsList>
      </div>
    </Tabs>
  )
}
