'use client'

import { noop, sum } from 'lodash-es'
import prettyBytes from 'pretty-bytes'
import useSWR from 'swr'
import { useQuery } from 'urql'

import { graphql } from '@/lib/gql/generates'
import { DiskUsage, DiskUsageStats } from '@/lib/gql/generates/graphql'
import { useHealth } from '@/lib/hooks/use-health'
import { useWorkers } from '@/lib/hooks/use-workers'
import { useMutation } from '@/lib/tabby/gql'
import { Button } from '@/components/ui/button'
import { IconRotate } from '@/components/ui/icons'
import { Input } from '@/components/ui/input'
import { Separator } from '@/components/ui/separator'
import { Skeleton } from '@/components/ui/skeleton'
import { CopyButton } from '@/components/copy-button'
import { ErrorView } from '@/components/error-view'
import LoadingWrapper from '@/components/loading-wrapper'

import { IngestionTable } from './ingestion-table'
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

const listIngestionStatus = graphql(/* GraphQL */ `
  query ingestionStatus($sources: [String!]) {
    ingestionStatus(sources: $sources) {
      source
      pending
      failed
      total
    }
  }
`)

function toBadgeString(str: string) {
  return encodeURIComponent(str.replaceAll('-', '--'))
}

export default function Workers() {
  const { data: healthInfo, error: healthError } = useHealth()
  const { data: workers, isLoading, error: workersError } = useWorkers()
  const [{ data: registrationTokenRes }, reexecuteQuery] = useQuery({
    query: getRegistrationTokenDocument
  })

  const [
    { data: ingestionStatusData, fetching: fetchingIngestion },
    reexecuteQueryIngestion
  ] = useQuery({
    query: listIngestionStatus
  })

  const resetRegistrationToken = useMutation(resetRegistrationTokenDocument, {
    onCompleted() {
      reexecuteQuery()
    }
  })

  useSWR(
    ingestionStatusData?.ingestionStatus?.length ? 'refresh_repos' : null,
    () => {
      reexecuteQueryIngestion()
    },
    {
      revalidateOnFocus: true,
      revalidateOnReconnect: true,
      revalidateOnMount: false,
      refreshInterval: 10 * 1000
    }
  )

  const error = healthError || workersError

  if (error) {
    return <ErrorView title={error?.message} />
  }

  if (!healthInfo) return

  return (
    <div className="flex w-full flex-col gap-3">
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
      <Usage />
      <Separator />
      <LoadingWrapper
        loading={isLoading}
        fallback={<Skeleton className="mt-3 h-32 w-full lg:w-2/3" />}
      >
        <>
          {!!registrationTokenRes?.registrationToken && (
            <div className="flex items-center gap-1 pt-2">
              Registration token:
              <Input
                className="max-w-[320px] font-mono"
                value={registrationTokenRes.registrationToken}
                onChange={noop}
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
          <div className="mt-3 flex flex-col gap-4 lg:flex-row lg:flex-wrap">
            {!!workers?.['COMPLETION'] && (
              <>
                {workers['COMPLETION'].map((worker, i) => {
                  return <WorkerCard key={i} {...worker} />
                })}
              </>
            )}
            {!!workers?.['CHAT'] && (
              <>
                {workers['CHAT'].map((worker, i) => {
                  return <WorkerCard key={i} {...worker} />
                })}
              </>
            )}
            {!!workers?.['EMBEDDING'] && (
              <>
                {workers['EMBEDDING'].map((worker, i) => {
                  return <WorkerCard key={i} {...worker} />
                })}
              </>
            )}
          </div>
        </>
      </LoadingWrapper>
      <LoadingWrapper
        loading={fetchingIngestion}
        fallback={<Skeleton className="mt-3 h-32 w-full lg:w-2/3" />}
      >
        {!!ingestionStatusData?.ingestionStatus?.length && (
          <>
            <Separator className="my-4" />
            <div className="font-bold">Documents Ingestion Status</div>
            <IngestionTable
              ingestionStatus={ingestionStatusData.ingestionStatus}
              className="mb-8 lg:w-[850px]"
            />
          </>
        )}
      </LoadingWrapper>
    </div>
  )
}

export const getDiskUsageStats = graphql(/* GraphQL */ `
  query GetDiskUsageStats {
    diskUsageStats {
      events {
        filepath
        sizeKb
      }
      indexedRepositories {
        filepath
        sizeKb
      }
      database {
        filepath
        sizeKb
      }
      models {
        filepath
        sizeKb
      }
    }
  }
`)

type UsageItem = {
  label: string
  key: keyof DiskUsageStats
  color: string
}

type UsageItemWithSize = UsageItem & { sizeKb: number }

const usageList: UsageItem[] = [
  {
    label: 'Model',
    key: 'models',
    color: '#0088FE'
  },
  {
    label: 'Indexing',
    key: 'indexedRepositories',
    color: '#00C49F'
  },
  {
    label: 'Event Logs',
    key: 'events',
    color: '#FF8042'
  },
  {
    label: 'Other',
    key: 'database',
    color: '#FFBB28'
  }
]

function Usage() {
  const [{ data, fetching }] = useQuery({
    query: getDiskUsageStats
  })

  let usageData: UsageItemWithSize[] = []
  let totalUsage: number = 0
  if (data) {
    usageData = usageList
      .map(usage => {
        if (!data.diskUsageStats[usage.key]) return null
        const diskUsage = data.diskUsageStats[usage.key] as DiskUsage
        return {
          ...usage,
          sizeKb: diskUsage.sizeKb
        }
      })
      .filter(usage => usage) as UsageItemWithSize[]
    totalUsage = sum(usageData.map(data => data.sizeKb))
  }

  return (
    <LoadingWrapper
      loading={fetching}
      fallback={<Skeleton className="mt-3 h-32 w-full lg:w-2/3" />}
    >
      <div className="flex flex-col gap-y-1.5 py-2">
        <div>
          <p className="mb-1 text-sm  text-muted-foreground">Disk Usage</p>
          <p className="text-3xl font-bold leading-none">
            {toBytes(totalUsage)}
          </p>
        </div>
        <div className="pt-3">
          <p className="mb-1 text-sm text-muted-foreground">
            Storage utilization by Type
          </p>
          <div className="flex flex-wrap gap-y-3">
            {usageData.map(usage => (
              <div className="flex w-1/2 pt-1 text-sm md:w-36" key={usage!.key}>
                <div
                  className="mr-3 mt-1 h-2 w-2 rounded-full"
                  style={{ backgroundColor: usage!.color }}
                />
                <div>
                  <p className="mb-1 leading-none">{usage!.label}</p>
                  <p className="text-card-foreground/70">
                    {toBytes(usage!.sizeKb)}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </LoadingWrapper>
  )
}

function toBytes(value: number) {
  return prettyBytes(value * 1000)
}
