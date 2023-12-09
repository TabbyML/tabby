'use client'

import {
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle
} from '@/components/ui/card'

import InvitationTable from './invitation-table'

export default function Team() {
  return (
    <div>
      <CardHeader>
        <CardTitle>Pending Invites</CardTitle>
      </CardHeader>
      <CardContent className="p-4">
        <InvitationTable />
      </CardContent>
      <CardFooter className="flex justify-end"></CardFooter>
    </div>
  )
}
