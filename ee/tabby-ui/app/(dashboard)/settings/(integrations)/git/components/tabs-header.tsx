'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'

import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'

export default function GitTabsHeader({ className }: { className?: string }) {
  const pathname = usePathname()
  const defualtValue = pathname.startsWith('/settings/git/generic')
    ? 'generic'
    : 'gitops'

  return (
    <Tabs defaultValue={defualtValue}>
      <div className="mb-4 flex">
        <TabsList className="grid grid-cols-2">
          <TabsTrigger value="generic" asChild>
            <Link href="/settings/git/generic">Generic Git Repositories</Link>
          </TabsTrigger>
          <TabsTrigger value="gitops" asChild>
            <Link href="/settings/git/gitops">Gitops</Link>
          </TabsTrigger>
        </TabsList>
      </div>
    </Tabs>
  )
}
