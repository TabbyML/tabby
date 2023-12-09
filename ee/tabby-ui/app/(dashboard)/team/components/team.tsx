'use client'

import { useRef } from 'react'

import {
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle
} from '@/components/ui/card'

import CreateInvitationForm from './create-invitation-form'
import InvitationTable from './invitation-table'

type InvitationTableHandle = React.ElementRef<typeof InvitationTable>

export default function Team() {
  const invitationTable = useRef<InvitationTableHandle>(null)
  return (
    <div>
      <CardHeader>
        <CardTitle>Pending Invites</CardTitle>
      </CardHeader>
      <CardContent className="p-4">
        <InvitationTable ref={invitationTable} />
      </CardContent>
      <CardFooter className="flex justify-end">
        <CreateInvitationForm
          onCreated={() => invitationTable.current?.changed()}
        />
      </CardFooter>
    </div>
  )
}
