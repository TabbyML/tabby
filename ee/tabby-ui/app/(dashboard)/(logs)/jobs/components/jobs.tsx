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
            <TableHead className="w-[20%]">Job</TableHead>
            <TableHead className="w-56 xl:w-auto">Recent Tasks</TableHead>
            <TableHead className="w-auto xl:w-[20%]">Last Run</TableHead>
            <TableHead className="w-[20%]">Job Runs</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {data?.jobs.map(jobName => {
            return <JobRow key={jobName} name={jobName} />
          })}
        </TableBody>
      </Table>
    </LoadingWrapper>
  )
}
