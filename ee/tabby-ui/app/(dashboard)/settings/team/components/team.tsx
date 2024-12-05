'use client'

import { useMe } from '@/lib/hooks/use-me'
import { CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import LoadingWrapper from '@/components/loading-wrapper'

import InvitationTable from './invitation-table'
import UsersTable from './user-table'

export default function Team() {
  const [{ data, fetching }] = useMe()
  return (
    <LoadingWrapper loading={fetching}>
      {data?.me.isAdmin && (
        <>
          <div>
            <CardHeader className="pl-0 pt-0">
              <CardTitle>Pending Invites</CardTitle>
            </CardHeader>
            <CardContent className="pl-0">
              <InvitationTable />
            </CardContent>
          </div>
          <div className="h-16" />
        </>
      )}
      <div>
        <CardHeader className="pl-0 pt-0">
          <CardTitle>Users</CardTitle>
        </CardHeader>
        <CardContent className="pl-0">
          <UsersTable />
        </CardContent>
      </div>
    </LoadingWrapper>
  )
}
