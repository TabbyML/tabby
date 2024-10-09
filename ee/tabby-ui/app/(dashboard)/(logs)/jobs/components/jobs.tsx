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

import { data } from '@patternfly/react-log-viewer/patternfly-docs/content/extensions/react-log-viewer/examples/./realTestData';
import { LogViewer, LogViewerSearch } from '@patternfly/react-log-viewer';
import { Toolbar, ToolbarContent, ToolbarItem } from '@patternfly/react-core';

const BasicSearchLogViewer = () => (
    <LogViewer
      data={data.data}
      toolbar={
        <Toolbar>
          <ToolbarContent>
            <ToolbarItem>
              <LogViewerSearch placeholder="Search value" />
            </ToolbarItem>
          </ToolbarContent>
        </Toolbar>
      }
    />
  );

const WithSearch = () => export default BasicSearchLogViewer;

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
            <TableHead className="w-[20%]">Job Runs-test</TableHead>
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
