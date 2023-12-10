'use client'

import { CardContent, CardHeader, CardTitle } from '@/components/ui/card'

import InvitationTable from './invitation-table'
import UsersTable from './user-table'

export default function Team() {
  return (
    <div className="xl:max-w-[750px]">
      <div>
        <CardHeader>
          <CardTitle>Invites</CardTitle>
        </CardHeader>
        <CardContent className="p-4">
          <InvitationTable />
        </CardContent>
      </div>
      <div>
        <CardHeader>
          <CardTitle>Users</CardTitle>
        </CardHeader>
        <CardContent className="p-4">
          <UsersTable />
        </CardContent>
      </div>
    </div>
  )
}
