'use client'

import React from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import humanizerDuration from 'humanize-duration'
import moment from 'moment'
import { useQuery } from 'urql'

import { listJobRuns } from '@/lib/tabby/query'
import {
  IconChevronLeft,
  IconClock,
  IconHistory,
  IconStopWatch
} from '@/components/ui/icons'
import { ListSkeleton } from '@/components/skeleton'

import { getJobDisplayName, getLabelByJobRun } from '../utils'
import '@patternfly/react-core/dist/styles/base.css'
import './styles.css'

import { LogViewer } from '@patternfly/react-log-viewer'

const BasicSearchLogViewer = ({ data }: { data?: string }) => {
  data = `2024-10-18T03:00:07.820391+00:00 [INFO]: Pulling source code for repository TabbyML/tabby
2024-10-18T03:00:07.820451+00:00 [INFO]: Building source code index: https://github.com/TabbyML/tabby
2024-10-18T03:00:08.333561+00:00 [INFO]: Processed 100/2019 files, updated 0 chunks
2024-10-18T03:00:08.349428+00:00 [INFO]: Processed 200/2019 files, updated 0 chunks
2024-10-18T03:00:08.376259+00:00 [INFO]: Processed 300/2019 files, updated 0 chunks
2024-10-18T03:00:08.398342+00:00 [INFO]: Processed 400/2019 files, updated 0 chunks
2024-10-18T03:00:08.413470+00:00 [INFO]: Processed 500/2019 files, updated 0 chunks
2024-10-18T03:00:08.452572+00:00 [INFO]: Processed 600/2019 files, updated 0 chunks
2024-10-18T03:00:08.473066+00:00 [INFO]: Processed 700/2019 files, updated 0 chunks
2024-10-18T03:00:08.488871+00:00 [INFO]: Processed 800/2019 files, updated 0 chunks
2024-10-18T03:00:08.523026+00:00 [INFO]: Processed 900/2019 files, updated 0 chunks
2024-10-18T03:00:09.425144+00:00 [INFO]: Processed 1000/2019 files, updated 0 chunks
2024-10-18T03:00:10.899999+00:00 [INFO]: Processed 1100/2019 files, updated 0 chunks
2024-10-18T03:00:13.602899+00:00 [INFO]: Processed 1200/2019 files, updated 0 chunks
2024-10-18T03:00:13.617632+00:00 [INFO]: Processed 1300/2019 files, updated 0 chunks
2024-10-18T03:00:13.635518+00:00 [INFO]: Processed 1400/2019 files, updated 0 chunks
2024-10-18T03:00:13.653326+00:00 [INFO]: Processed 1500/2019 files, updated 0 chunks
2024-10-18T03:00:13.667583+00:00 [INFO]: Processed 1600/2019 files, updated 0 chunks
2024-10-18T03:00:13.683357+00:00 [INFO]: Processed 1700/2019 files, updated 0 chunks
2024-10-18T03:00:13.705367+00:00 [INFO]: Processed 1800/2019 files, updated 0 chunks
2024-10-18T03:00:13.726845+00:00 [INFO]: Processed 1900/2019 files, updated 0 chunks
2024-10-18T03:00:13.742781+00:00 [INFO]: Processed 2000/2019 files, updated 0 chunks
2024-10-18T03:00:13.751033+00:00 [INFO]: Processed 2019/2019 files, updated 0 chunks
2024-10-18T03:00:13.834474+00:00 [INFO]: Finished garbage collection for code index: 981 items kept, 0 items removed
2024-10-18T03:00:13.848343+00:00 [INFO]: Indexing documents for repository TabbyML/tabby
2024-10-18T03:00:28.735898+00:00 [INFO]: 100 docs seen, 1 docs updated
2024-10-18T03:00:30.199363+00:00 [INFO]: 200 docs seen, 1 docs updated
2024-10-18T03:00:31.689048+00:00 [INFO]: 300 docs seen, 1 docs updated
2024-10-18T03:00:33.608242+00:00 [INFO]: 400 docs seen, 1 docs updated
2024-10-18T03:00:35.153361+00:00 [INFO]: 500 docs seen, 1 docs updated
2024-10-18T03:00:36.619432+00:00 [INFO]: 600 docs seen, 1 docs updated
2024-10-18T03:00:38.493263+00:00 [INFO]: 700 docs seen, 1 docs updated
2024-10-18T03:00:39.923201+00:00 [INFO]: 800 docs seen, 1 docs updated
2024-10-18T03:00:41.218285+00:00 [INFO]: 900 docs seen, 1 docs updated
2024-10-18T03:00:43.128538+00:00 [INFO]: 1000 docs seen, 1 docs updated
2024-10-18T03:00:44.606865+00:00 [INFO]: 1100 docs seen, 1 docs updated
2024-10-18T03:00:45.976754+00:00 [INFO]: 1200 docs seen, 1 docs updated
2024-10-18T03:00:47.872129+00:00 [INFO]: 1300 docs seen, 1 docs updated
2024-10-18T03:00:49.313460+00:00 [INFO]: 1400 docs seen, 1 docs updated
2024-10-18T03:00:50.811361+00:00 [INFO]: 1500 docs seen, 1 docs updated
2024-10-18T03:00:52.691501+00:00 [INFO]: 1600 docs seen, 1 docs updated
2024-10-18T03:00:54.168103+00:00 [INFO]: 1700 docs seen, 1 docs updated
2024-10-18T03:00:55.600957+00:00 [INFO]: 1800 docs seen, 1 docs updated
2024-10-18T03:00:57.575628+00:00 [INFO]: 1900 docs seen, 1 docs updated
2024-10-18T03:00:59.229032+00:00 [INFO]: 2000 docs seen, 1 docs updated
2024-10-18T03:01:00.831301+00:00 [INFO]: 2100 docs seen, 1 docs updated
2024-10-18T03:01:02.918650+00:00 [INFO]: 2200 docs seen, 1 docs updated
2024-10-18T03:01:04.455951+00:00 [INFO]: 2300 docs seen, 1 docs updated
2024-10-18T03:01:06.073000+00:00 [INFO]: 2400 docs seen, 1 docs updated
2024-10-18T03:01:08.241701+00:00 [INFO]: 2500 docs seen, 1 docs updated
2024-10-18T03:01:09.699005+00:00 [INFO]: 2600 docs seen, 1 docs updated
2024-10-18T03:01:11.215524+00:00 [INFO]: 2700 docs seen, 1 docs updated
2024-10-18T03:01:13.224199+00:00 [INFO]: 2800 docs seen, 1 docs updated
2024-10-18T03:01:14.854180+00:00 [INFO]: 2900 docs seen, 1 docs updated
2024-10-18T03:01:16.561093+00:00 [INFO]: 3000 docs seen, 1 docs updated
2024-10-18T03:01:18.657958+00:00 [INFO]: 3100 docs seen, 1 docs updated
2024-10-18T03:01:20.196735+00:00 [INFO]: 3200 docs seen, 1 docs updated
2024-10-18T03:01:20.198013+00:00 [INFO]: 3204 docs seen, 1 docs updated
2024-10-18T03:01:20.394055+00:00 [INFO]: Job completed successfully`
  // debugger
  // data?.repeat(10);
  // console.log(data);

  return (
    <LogViewer
      data={data}
      hasLineNumbers={false}
      height={'72vh'}
    />
  )
}

export default function JobRunDetail() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const id = searchParams.get('id')
  const [{ data, error, fetching }, reexecuteQuery] = useQuery({
    query: listJobRuns,
    variables: { ids: [id as string] },
    pause: !id
  })

  const currentNode = data?.jobRuns?.edges?.[0]?.node

  const stateLabel = getLabelByJobRun(currentNode)
  const isPending =
    (stateLabel === 'Pending' || stateLabel === 'Running') &&
    !currentNode?.stdout

  React.useEffect(() => {
    let timer: number
    if (currentNode?.createdAt && !currentNode?.finishedAt) {
      timer = window.setTimeout(() => {
        reexecuteQuery()
      }, 5000)
    }

    return () => {
      if (timer) {
        clearInterval(timer)
      }
    }
  }, [currentNode])

  return (
    <>
      {fetching ? (
        <ListSkeleton />
      ) : (
        <div className="flex flex-1 flex-col items-stretch gap-2">
          {currentNode && (
            <>
              <div
                onClick={() => router.back()}
                className="-ml-1 flex cursor-pointer items-center transition-opacity hover:opacity-60"
              >
                <IconChevronLeft className="mr-1 h-6 w-6" />
                <h2 className="scroll-m-20 text-3xl font-bold tracking-tight first:mt-0">
                  {getJobDisplayName(currentNode.job)}
                </h2>
              </div>
              <div className="mb-2 flex gap-x-5 text-sm text-muted-foreground lg:gap-x-10">
                <div className="flex items-center gap-1">
                  <IconStopWatch />
                  <p>State: {stateLabel}</p>
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

                {currentNode.startedAt && (
                  <div className="flex items-center gap-1">
                    <IconClock />
                    <p>
                      Started:{' '}
                      {moment(currentNode.startedAt).format('YYYY-MM-DD HH:mm')}
                    </p>
                  </div>
                )}

                {currentNode.createdAt && currentNode.finishedAt && (
                  <div className="flex items-center gap-1">
                    <IconHistory />
                    <p>
                      Duration:{' '}
                      {humanizerDuration(
                        moment
                          .duration(
                            moment(currentNode.finishedAt).diff(
                              currentNode.startedAt
                            )
                          )
                          .asMilliseconds()
                      )}
                    </p>
                  </div>
                )}
              </div>
              <div className="flex flex-1 flex-col">
                <BasicSearchLogViewer data={currentNode?.stdout} />
              </div>
            </>
          )}
        </div>
      )}
    </>
  )
}
