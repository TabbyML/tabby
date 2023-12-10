'use client'

import { CardContent, CardHeader, CardTitle } from '@/components/ui/card'

import InvitationTable from './invitation-table'
import UsersTable from './user-table'

export default function Team() {
  return (
    <div>
      <div>
        <CardHeader>
          <CardTitle>Pending Invites</CardTitle>
        </CardHeader>
        <CardContent className="p-4">
          <InvitationTable />
        </CardContent>
      </div>
      <div className="h-16" />
      <div>
        <CardHeader>
          <CardTitle>Members</CardTitle>
        </CardHeader>
        <CardContent className="p-4">
          <UsersTable />
        </CardContent>
      </div>
    </div>
  )
}
