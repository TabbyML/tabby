'use client'

import React from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import moment from 'moment'

import { Button } from '@/components/ui/button'
import { CardContent, CardTitle } from '@/components/ui/card'
import {
  IconChevronLeft,
  IconClock,
  IconStopWatch
} from '@/components/ui/icons'
import LoadingWrapper from '@/components/loading-wrapper'

import CreateUserGroupDialog from '../../components/create-user-group'

export default function JobRunDetail() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const id = searchParams.get('id')
  // const [{ data, error, fetching }, reexecuteQuery] = useQuery({
  //   query: listJobRuns,
  //   variables: { ids: [id as string] },
  //   pause: !id
  // })

  // const currentNode = data?.jobRuns?.edges?.[0]?.node
  const currentNode: any = {
    createdAt: '2024-09-05T09:30:00.000Z',
    updatedAt: '2024-09-05T09:30:00.000Z'
  }
  const fetching = false

  return (
    <LoadingWrapper loading={fetching}>
      <CardTitle className="flex items-center gap-4">
        <div className="-ml-2.5 flex items-center">
          <Button
            onClick={() => router.back()}
            variant={'ghost'}
            className="h-6 px-1"
          >
            <IconChevronLeft className="h-5 w-5" />
          </Button>
          <span className="ml-1">Group Name</span>
        </div>
      </CardTitle>
      <CardContent className="mt-4">
        <div className="flex gap-x-5 text-sm text-muted-foreground lg:gap-x-10">
          <div className="flex items-center gap-1">
            <IconStopWatch />
            <p>State: Active</p>
          </div>

          {currentNode.createdAt && (
            <div className="flex items-center gap-1">
              <IconClock />
              <p>
                Created:{' '}
                {moment(currentNode.createdAt).format('YYYY-MM-DD HH:mm')}
              </p>
            </div>
          )}

          {currentNode.updatedAt && (
            <div className="flex items-center gap-1">
              <IconClock />
              <p>
                Updated:{' '}
                {moment(currentNode.updatedAt).format('YYYY-MM-DD HH:mm')}
              </p>
            </div>
          )}
        </div>
      </CardContent>
      <CardContent className="mt-8">
        <CreateUserGroupDialog
          isNew={false}
          onSubmit={() => {
            console.log('create')
          }}
        />
      </CardContent>
    </LoadingWrapper>
  )
}
