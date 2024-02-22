'use client'

import { IconCheck } from '@/components/ui/icons'
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
        <TableRow className="hidden">
          <TableHead className="w-[40%]"></TableHead>
          <TableHead className="w-[20%]"></TableHead>
          <TableHead className="w-[20%]"></TableHead>
          <TableHead className="w-[20%]"></TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        <TableRow>
          <TableCell
            colSpan={4}
            className="bg-accent text-left text-accent-foreground"
          >
            Change Management
          </TableCell>
        </TableRow>
        <TableRow>
          <TableCell className="text-left">User count</TableCell>
          <TableCell>Up to 20</TableCell>
          <TableCell>Credit Card</TableCell>
          <TableCell>$250.00</TableCell>
        </TableRow>
        <TableRow>
          <TableCell className="text-left">State-based change</TableCell>
          <TableCell>
            <div className="flex justify-center">
              {/* todo switch to another icon which can set strokeWidth */}
              <IconCheck style={{ strokeWidth: 2 }} />
            </div>
          </TableCell>
          <TableCell>
            <div className="flex justify-center">
              <IconCheck />
            </div>
          </TableCell>
          <TableCell>
            <div className="flex justify-center">
              <IconCheck />
            </div>
          </TableCell>
        </TableRow>
        <TableRow>
          <TableCell
            colSpan={4}
            className="bg-accent text-left text-accent-foreground"
          >
            Security
          </TableCell>
        </TableRow>
        <TableRow>
          <TableCell className="text-left">User count</TableCell>
          <TableCell>Up to 20</TableCell>
          <TableCell>Credit Card</TableCell>
          <TableCell>$250.00</TableCell>
        </TableRow>
      </TableBody>
    </Table>
  )
}
