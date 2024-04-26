'use client'

import bytes from 'bytes'
import { noop, sum } from 'lodash-es'
import { useTheme } from 'next-themes'
import { Cell, Label, Pie, PieChart, ResponsiveContainer } from 'recharts'
import { useQuery } from 'urql'

import { graphql } from '@/lib/gql/generates'
import {
  DiskUsage,
  DiskUsageStats,
  WorkerKind
} from '@/lib/gql/generates/graphql'
import { useHealth } from '@/lib/hooks/use-health'
import { useWorkers } from '@/lib/hooks/use-workers'
import { useMutation } from '@/lib/tabby/gql'
import { Button } from '@/components/ui/button'
import { IconRotate } from '@/components/ui/icons'
import { Input } from '@/components/ui/input'
import { Separator } from '@/components/ui/separator'
import { CopyButton } from '@/components/copy-button'
import LoadingWrapper from '@/components/loading-wrapper'

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
  const [{ data: registrationTokenRes }, reexecuteQuery] = useQuery({
    query: getRegistrationTokenDocument
  })

  const resetRegistrationToken = useMutation(resetRegistrationTokenDocument, {
    onCompleted() {
      reexecuteQuery()
    }
  })

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
      {!!registrationTokenRes?.registrationToken && (
        <div className="flex items-center gap-1">
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

      <Separator className="mt-6" />
      <Usage />
    </div>
  )
}

export const getDiskUsageStats = graphql(/* GraphQL */ `
  query GetDiskUsageStats {
    diskUsageStats {
      events {
        filePaths
        size
      }
      indexedRepositories {
        filePaths
        size
      }
      database {
        filePaths
        size
      }
      models {
        filePaths
        size
      }
    }
  }
`)

type UsageItem = {
  label: string
  key: keyof DiskUsageStats
  color: string
}

type UsageItemWithSize = UsageItem & { size: number }

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
          size: diskUsage.size
        }
      })
      .filter(usage => usage) as UsageItemWithSize[]
    totalUsage = sum(usageData.map(data => data.size))
  }

  return (
    <LoadingWrapper loading={fetching} fallback={<></>}>
      <>
        <div>
          <p className="font-bold">Disk Usage</p>
          <p className="text-sm text-muted-foreground">
            Storage utilization by Type
          </p>
        </div>
        <div className="flex flex-col items-center gap-x-3 md:flex-row">
          <ResponsiveContainer width={230} height={220}>
            <PieChart>
              <Pie
                data={usageData}
                dataKey="size"
                cx={110}
                cy={100}
                innerRadius={70}
                outerRadius={90}
                stroke="none"
              >
                {usageData.map(entry => (
                  <Cell key={entry.key} fill={entry.color} />
                ))}
                <Label
                  content={<CustomLabel totalUsage={totalUsage} />}
                  position="center"
                />
              </Pie>
            </PieChart>
          </ResponsiveContainer>

          <div className="flex w-full flex-col gap-y-2 md:ml-10 md:w-auto">
            {usageData.map(usage => (
              <div
                className="flex cursor-default items-center justify-between text-xs"
                key={usage!.key}
              >
                <div className="flex w-40 items-center">
                  <div
                    className="mr-1.5 h-3 w-3 rounded"
                    style={{ backgroundColor: usage!.color }}
                  />
                  <p className="font-semibold">{usage!.label}</p>
                </div>
                <p>{toBytes(usage!.size)}</p>
              </div>
            ))}
          </div>
        </div>
      </>
    </LoadingWrapper>
  )
}

function CustomLabel({
  viewBox,
  totalUsage
}: {
  viewBox?: {
    cx: number
    cy: number
  }
  totalUsage: number
}) {
  const { theme } = useTheme()
  if (!viewBox) return
  const { cx, cy } = viewBox
  return (
    <g>
      <text
        x={cx}
        y={cy - 20}
        textAnchor="middle"
        dominantBaseline="central"
        alignmentBaseline="middle"
        fill={theme === 'dark' ? '#FDFDFD' : '#030302'}
        className="text-sm"
      >
        Total Usage
      </text>
      <text
        x={cx}
        y={cy + 13}
        textAnchor="middle"
        dominantBaseline="central"
        alignmentBaseline="middle"
        fill={theme === 'dark' ? '#FDFDFD' : '#030302'}
        className="text-lg font-semibold"
      >
        {toBytes(totalUsage)}
      </text>
    </g>
  )
}

function toBytes(value: number) {
  return bytes(value * 1024, { unitSeparator: ' ' })
}
