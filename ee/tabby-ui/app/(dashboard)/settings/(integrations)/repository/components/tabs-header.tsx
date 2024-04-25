'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'

import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'

export default function GitTabsHeader() {
  const pathname = usePathname()
  const defualtValue = pathname.indexOf('github') >= 0 ? 'github' : 'git'

  return (
    <Tabs defaultValue={defualtValue}>
      <div className="sticky top-0 mb-4 flex">
        <TabsList className="grid grid-cols-2">
          <TabsTrigger value="git" asChild>
            <Link href="git">Git</Link>
          </TabsTrigger>
          <TabsTrigger value="github" asChild>
            <Link href="github">Github</Link>
          </TabsTrigger>
        </TabsList>
      </div>
    </Tabs>
  )
}
