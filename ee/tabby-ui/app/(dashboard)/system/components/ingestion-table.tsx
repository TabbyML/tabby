import { useMemo, useState } from 'react'

import { IngestionStats } from '@/lib/gql/generates/graphql'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from '@/components/ui/table'
import { QuickNavPagination } from '@/components/quick-nav-pagination'

const PAGE_SIZE = 10

export function IngestionTable({
  ingestionStatus,
  className
}: {
  ingestionStatus: IngestionStats[] | undefined
  className?: string
}) {
  const [page, setPage] = useState(1)
  const nodes = useMemo(() => {
    return (
      ingestionStatus?.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE) ?? []
    )
  }, [ingestionStatus, page])

  return (
    <div className={className}>
      <Table>
        <TableHeader>
          <TableRow className="border-b">
            <TableHead className="px-6 py-3 font-medium text-muted-foreground">
              Source Name
            </TableHead>
            <TableHead className="w-[100px] px-6 py-3 text-center font-medium text-muted-foreground">
              Pending
            </TableHead>
            <TableHead className="w-[100px] px-6 py-3 text-center font-medium text-muted-foreground">
              Failed
            </TableHead>
            <TableHead className="w-[100px] px-6 py-3 text-center font-medium text-muted-foreground">
              Total
            </TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {nodes.map(x => {
            return (
              <TableRow key={x.source} className="border-b">
                <TableCell className="px-6 py-3">{x.source}</TableCell>
                <TableCell className="px-6 py-3 text-center">
                  {x.pending}
                </TableCell>
                <TableCell className="px-6 py-3 text-center">
                  {x.failed}
                </TableCell>
                <TableCell className="px-6 py-3 text-center">
                  {x.total}
                </TableCell>
              </TableRow>
            )
          })}
        </TableBody>
      </Table>
      <QuickNavPagination
        className="mt-2 flex justify-end"
        page={page}
        pageSize={PAGE_SIZE}
        showQuickJumper
        totalCount={ingestionStatus?.length ?? 0}
        onChange={(page: number) => {
          setPage(page)
        }}
      />
    </div>
  )
}
