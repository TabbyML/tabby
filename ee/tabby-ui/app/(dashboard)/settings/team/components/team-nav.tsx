'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'

import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'

export function TeamNav() {
  const pathname = usePathname()
  const isGroupsPath = pathname.includes('/team/groups')

  return (
    <Tabs value={isGroupsPath ? 'groups' : 'members'} className="mb-8">
      <TabsList>
        <Link href="/settings/team">
          <TabsTrigger value="members">Users</TabsTrigger>
        </Link>
        <Link href="/settings/team/groups">
          <TabsTrigger value="groups">Groups</TabsTrigger>
        </Link>
      </TabsList>
    </Tabs>
  )
}
