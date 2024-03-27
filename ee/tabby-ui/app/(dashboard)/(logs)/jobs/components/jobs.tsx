'use client'

import { Metadata } from 'next'
import { useQuery } from 'urql'

import { listJobs } from '@/lib/tabby/query'

import {
  Table,
  TableBody,
  TableHead,
  TableHeader,
  TableRow
} from '@/components/ui/table'
import LoadingWrapper from '@/components/loading-wrapper'
import JobRow from './job-row'

export const metadata: Metadata = {
  title: 'Jobs'
}

export default function JobRunsPage() {
  const [{ data, fetching }] = useQuery({
    query: listJobs
  })

  return (
    <LoadingWrapper loading={fetching}>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-[20%]">Name</TableHead>
            <TableHead>Recent Tasks</TableHead>
            <TableHead className="w-[20%]">Last Run</TableHead>
            <TableHead className="w-[30%]">Aggregate Runs</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {data?.jobs.map(jobName => {
            return (
              <JobRow key={jobName} name={jobName} />
            )
          })}
        </TableBody>
      </Table>
    </LoadingWrapper>
  )
}
