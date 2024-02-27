'use client'

import { CardContent, CardHeader, CardTitle } from '@/components/ui/card'

import InvitationTable from './invitation-table'
import UsersTable from './user-table'

export default function Team() {
  return (
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
      <div>
        <CardHeader className="pl-0 pt-0">
          <CardTitle>Members</CardTitle>
        </CardHeader>
        <CardContent className="pl-0">
          <UsersTable />
        </CardContent>
      </div>
    </>
  )
}
