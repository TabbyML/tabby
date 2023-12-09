'use client'

import { CardContent, CardHeader, CardTitle } from '@/components/ui/card'

import InvitationTable from './invitation-table'

export default function Team() {
  return (
    <div>
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
        <CardContent className="p-4"></CardContent>
      </div>
    </div>
  )
}
