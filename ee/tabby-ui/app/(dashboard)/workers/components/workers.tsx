'use client'

import { graphql } from '@/lib/gql/generates'
import { WorkerKind } from '@/lib/gql/generates/graphql'
import { useHealth } from '@/lib/hooks/use-health'
import { useWorkers } from '@/lib/hooks/use-workers'
import { useAuthenticatedGraphQLQuery, useGraphQLForm } from '@/lib/tabby/gql'
import { CopyButton } from '@/components/copy-button'

import WorkerCard from './worker-card'
import { Button } from '@/components/ui/button'
import { IconRefresh } from '@/components/ui/icons'

const getRegistrationTokenDocument = graphql(/* GraphQL */ `
  query GetRegistrationToken {
    registrationToken
  }
`)

const resetRegistrationTokenDocument = graphql(/* GraphQL */ `
  mutation ResetRegistrationToken {
    resetRegistrationToken
  }
`)

export default function Workers() {
  const { data: healthInfo } = useHealth()
  const workers = useWorkers()
  const { data: registrationTokenRes, mutate } = useAuthenticatedGraphQLQuery(
    getRegistrationTokenDocument
  )

  const { onSubmit: resetRegistrationToken } = useGraphQLForm(resetRegistrationTokenDocument, {
    onSuccess: () => mutate(),
  })

  if (!healthInfo) return

  return (
    <div className="p-4 lg:p-16 flex w-full flex-col gap-3">
      {!!registrationTokenRes?.registrationToken && (
        <div className="flex items-center gap-1">
          Registeration token:
          <code className="rounded-lg text-sm text-red-600 p-1">
            {registrationTokenRes.registrationToken}
          </code>
          <Button
            title="Reset"
            size="icon"
            variant="ghost"
            className='hover:bg-destructive/90 hover:text-destructive-foreground'
            onClick={() => resetRegistrationToken()}><IconRefresh /></Button>
          <CopyButton value={registrationTokenRes.registrationToken} />
        </div>
      )}

      <div className="mt-4 flex flex-col gap-4 lg:flex-row lg:flex-wrap">
        {!!workers?.[WorkerKind.Completion] && (
          <>
            {workers[WorkerKind.Completion].map((worker, i) => {
              return <WorkerCard key={i} {...worker} />
            })}
          </>
        )}
        {!!workers?.[WorkerKind.Chat] && (
          <>
            {workers[WorkerKind.Chat].map((worker, i) => {
              return <WorkerCard key={i} {...worker} />
            })}
          </>
        )}
        <WorkerCard
          addr="localhost"
          name="Code Search Index"
          kind="INDEX"
          arch=""
          device={healthInfo.device}
          cudaDevices={healthInfo.cuda_devices}
          cpuCount={healthInfo.cpu_count}
          cpuInfo={healthInfo.cpu_info}
        />
      </div>
    </div>
  )
}
