'use client'

import React, { useEffect, useState } from 'react'
import { toast } from 'sonner'

import { DEFAULT_PAGE_SIZE } from '@/lib/constants'
import { useMutation } from '@/lib/tabby/gql'
import { Switch } from '@/components/ui/switch'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from '@/components/ui/table'
import LoadingWrapper from '@/components/loading-wrapper'

import { JobInfoView } from '../../components/job-trigger'
import { triggerJobRunMutation } from '../../query'

const PAGE_SIZE = DEFAULT_PAGE_SIZE

export default function PresetDocTable() {
  // FIXME for mock
  const [fetching, setFetching] = useState(true)
  const [data, setData] = useState<any[]>([])

  // FIXME
  useEffect(() => {
    window.setTimeout(() => {
      setFetching(false)
      setData([
        {
          id: '1',
          node: {
            name: 'Tabby',
            url: 'https://tabby.tabbyml.com'
          }
        },
        {
          id: '2',
          node: {
            name: 'Skypilot',
            url: 'https://skypilot.readthedocs.io/'
          }
        }
      ])
    }, 2000)
  }, [])

  const triggerJobRun = useMutation(triggerJobRunMutation)
  const handleTriggerJobRun = (command: string) => {
    return triggerJobRun({ command }).then(res => {
      if (res?.data?.triggerJobRun) {
        toast.success(
          'The job has been triggered successfully, it may take a few minutes to process.'
        )
        // reexecuteQuery()
      } else {
        toast.error(res?.error?.message || 'Failed to trigger job')
      }
    })
  }

  return (
    <LoadingWrapper loading={fetching}>
      <Table className="table-fixed border-b">
        <TableHeader>
          <TableRow>
            <TableHead className="w-[30%]">Name</TableHead>
            <TableHead className="w-[40%]">URL</TableHead>
            <TableHead>Job</TableHead>
            <TableHead className="w-[100px] text-right">Active</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {!data?.length && !fetching ? (
            <TableRow>
              <TableCell colSpan={4} className="h-[100px] text-center">
                No Data
              </TableCell>
            </TableRow>
          ) : (
            <>
              {data?.map(x => {
                return (
                  <TableRow key={x.node.id}>
                    <TableCell className="break-all lg:break-words">
                      {x.node.name}
                    </TableCell>
                    <TableCell className="break-all lg:break-words">
                      {x.node.url}
                    </TableCell>
                    <TableCell>
                      <JobInfoView
                        jobInfo={{ command: 'test' }}
                        onTrigger={async () => {
                          // handleTriggerJobRun(x.node.jobInfo.command)
                        }}
                      />
                    </TableCell>
                    <TableCell className="text-right">
                      <Switch />
                    </TableCell>
                  </TableRow>
                )
              })}
            </>
          )}
        </TableBody>
      </Table>
    </LoadingWrapper>
  )
}
