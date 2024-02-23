'use client'

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from '@/components/ui/table'

export const LicenseTable = () => {
  return (
    <Table className="border text-center">
      <TableHeader>
        <TableRow>
          <TableHead className="w-[40%]"></TableHead>
          <TableHead className="w-[20%] text-center">Free</TableHead>
          <TableHead className="w-[20%] text-center">Team</TableHead>
          <TableHead className="w-[20%] text-center">Enterprise</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        <TableRow>
          <TableCell
            colSpan={4}
            className="bg-accent text-left text-accent-foreground"
          >
            Member Management
          </TableCell>
        </TableRow>
        <TableRow>
          <TableCell className="text-left">Seat count</TableCell>
          <TableCell>1</TableCell>
          <TableCell>Up to 10</TableCell>
          <TableCell>Unlimited</TableCell>
        </TableRow>
      </TableBody>
    </Table>
  )
}
