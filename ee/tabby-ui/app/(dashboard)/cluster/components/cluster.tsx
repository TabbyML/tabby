'use client'

import { graphql } from '@/lib/gql/generates'
import { WorkerKind } from '@/lib/gql/generates/graphql'
import { useHealth } from '@/lib/hooks/use-health'
import { useWorkers } from '@/lib/hooks/use-workers'
import { useAuthenticatedGraphQLQuery, useMutation } from '@/lib/tabby/gql'
import { Button } from '@/components/ui/button'
import { IconRotate } from '@/components/ui/icons'
import { Input } from '@/components/ui/input'
import { Separator } from '@/components/ui/separator'
import { CopyButton } from '@/components/copy-button'

import WorkerCard from './worker-card'

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

function toBadgeString(str: string) {
  return encodeURIComponent(str.replaceAll('-', '--'))
}

export default function Workers() {
  const { data: healthInfo } = useHealth()
  const workers = useWorkers()
  const { data: registrationTokenRes, mutate } = useAuthenticatedGraphQLQuery(
    getRegistrationTokenDocument
  )

  const resetRegistrationToken = useMutation(resetRegistrationTokenDocument, {
    onCompleted() {
      mutate()
    }
  })

  if (!healthInfo) return

  return (
    <div className="flex w-full flex-col gap-3 p-4 lg:p-16">
      <h1>
        <span className="font-bold">Congratulations</span>, your tabby instance
        is up!
      </h1>
      <span className="flex flex-wrap gap-1">
        <a
          target="_blank"
          href={`https://github.com/TabbyML/tabby/releases/tag/${healthInfo.version.git_describe}`}
        >
          <img
            src={`https://img.shields.io/badge/version-${toBadgeString(
              healthInfo.version.git_describe
            )}-green`}
          />
        </a>
      </span>
      <Separator />
      {!!registrationTokenRes?.registrationToken && (
        <div className="flex items-center gap-1">
          Registration token:
          <Input
            className="max-w-[320px] font-mono text-red-600"
            value={registrationTokenRes.registrationToken}
          />
          <Button
            title="Rotate"
            size="icon"
            variant="hover-destructive"
            onClick={() => resetRegistrationToken()}
          >
            <IconRotate />
          </Button>
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
