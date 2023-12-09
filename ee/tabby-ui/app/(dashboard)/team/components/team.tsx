'use client'

import moment from "moment"
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableFooter,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { graphql } from "@/lib/gql/generates"
import { gqlClient, useAuthenticatedGraphQLQuery, useGraphQLForm } from "@/lib/tabby/gql"
import { Badge } from "@/components/ui/badge"

import { CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import CreateInvitationForm from "./create-invitation-form"
import { ListInvitationsQuery } from "@/lib/gql/generates/graphql"
import { CopyButton } from "@/components/copy-button"
import { Button } from "@/components/ui/button"
import { IconTrash } from "@/components/ui/icons"
import { Separator } from "@/components/ui/separator"
import InvitationTable from "./invitation-table"
import { useRef } from "react"


type InvitationTableHandle = React.ElementRef<typeof InvitationTable>;

export default function Team() {
  const invitationTable = useRef<InvitationTableHandle>(null)
  return <div>
    <CardHeader>
      <CardTitle>Pending Invites</CardTitle>
    </CardHeader>
    <CardContent className="p-4">
      <InvitationTable ref={invitationTable} />
    </CardContent>
    <CardFooter className="flex justify-end">
      <CreateInvitationForm onCreated={() => invitationTable.current?.changed()} />
    </CardFooter>
  </div>
}