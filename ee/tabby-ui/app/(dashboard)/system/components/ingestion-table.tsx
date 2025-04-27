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
          <TableRow>
            <TableHead>Ingestion Group</TableHead>
            <TableHead className="w-[90px] text-center">Pending</TableHead>
            <TableHead className="w-[90px] text-center">Failed</TableHead>
            <TableHead className="w-[90px] text-center">Total</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {nodes.map(x => {
            return (
              <TableRow key={x.source}>
                <TableCell>{x.source.replace(/^ingested:/, '')}</TableCell>
                <TableCell className="text-center">{x.pending}</TableCell>
                <TableCell className="text-center">{x.failed}</TableCell>
                <TableCell className="text-center">{x.total}</TableCell>
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
