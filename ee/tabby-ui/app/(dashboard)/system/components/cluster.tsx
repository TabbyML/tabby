'use client'

import { ReactNode, useEffect, useMemo, useRef, useState } from 'react'
import { isNil, noop, sum } from 'lodash-es'
import prettyBytes from 'pretty-bytes'
import { useQuery } from 'urql'

import { graphql } from '@/lib/gql/generates'
import {
  DiskUsage,
  DiskUsageStats,
  ModelHealthBackend
} from '@/lib/gql/generates/graphql'
import { useHealth } from '@/lib/hooks/use-health'
import { useWorkers } from '@/lib/hooks/use-workers'
import { useMutation } from '@/lib/tabby/gql'
import { Button } from '@/components/ui/button'
import { IconRotate } from '@/components/ui/icons'
import { Input } from '@/components/ui/input'
import { Separator } from '@/components/ui/separator'
import { Skeleton } from '@/components/ui/skeleton'
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

const testModelConnectionQuery = graphql(/* GraphQL */ `
  query TestModelConnection($backend: ModelHealthBackend!) {
    testModelConnection(backend: $backend) {
      latencyMs
    }
  }
`)

function toBadgeString(str: string) {
  return encodeURIComponent(str.replaceAll('-', '--'))
}

export default function Workers() {
  const { data: healthInfo } = useHealth()
  const { data: workers, fetching } = useWorkers()
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
      <Usage />
      <div className="mt-6 font-semibold">Workers</div>
      <Separator />
      <LoadingWrapper
        loading={fetching}
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
        </>
      </LoadingWrapper>
      <div className="mt-6 font-semibold">Model connection</div>
      <Separator />
      <div className="gap-sm grid grid-cols-3 gap-2 overflow-hidden md:grid-cols-4">
        <ModelConnection backend={ModelHealthBackend.Completion} />
        <ModelConnection backend={ModelHealthBackend.Chat} />
        <ModelConnection backend={ModelHealthBackend.Embedding} />
      </div>
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

function ModelConnection({ backend }: { backend: ModelHealthBackend }) {
  const [progress, setProgress] = useState(0)
  const [{ data, fetching, error, stale }, reexecuteQuery] = useQuery({
    query: testModelConnectionQuery,
    variables: {
      backend
    }
  })

  const latency = data?.testModelConnection?.latencyMs
  const timer = useRef(0)
  const isLoading = fetching || stale

  const color = useMemo(() => {
    if (isLoading) {
      return '#007bff'
    }

    if (!isNil(latency)) {
      return latency > 3000 ? '#ffbb28' : '#16a34a'
    }

    return '#dc2626'
  }, [progress, error, latency, isLoading])

  useEffect(() => {
    if (isLoading) {
      setProgress(0)
      timer.current = window.setInterval(() => {
        setProgress(p => Math.min(p + 1, 90))
      }, 64)
    } else {
      setProgress(100)
    }

    return () => {
      window.clearInterval(timer.current)
    }
  }, [isLoading])

  return (
    <div>
      <CircularProgress
        progress={progress}
        size={168}
        strokeWidth={10}
        className="my-4"
        color={color}
      >
        <div className="flex flex-1 flex-col items-center pt-3">
          <ModelConnectionTitle backend={backend} />
          <div className="mb-4 mt-2 flex items-center gap-0.5 text-sm text-muted-foreground">
            <span>Latency:</span>
            {isLoading ? (
              <Skeleton className="h-3 w-10 rounded-sm" />
            ) : (
              `${data?.testModelConnection?.latencyMs}ms`
            )}
          </div>
          <Button
            size="sm"
            disabled={isLoading}
            onClick={e => reexecuteQuery()}
            className="gap-1"
            variant="outline"
          >
            <span className="text-sm">Retry</span>
          </Button>
        </div>
      </CircularProgress>
    </div>
  )
}

function ModelConnectionTitle({ backend }: { backend: ModelHealthBackend }) {
  let icon: ReactNode = null
  let title: string = ''
  let className = 'h-5 w-5'
  switch (backend) {
    case ModelHealthBackend.Completion:
      icon = (
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className={className}
          width="24"
          height="24"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
          <polyline points="14 2 14 8 20 8" />
          <path d="m10 13-2 2 2 2" />
          <path d="m14 17 2-2-2-2" />
        </svg>
      )
      title = 'Completion'
      break
    case ModelHealthBackend.Chat:
      icon = (
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className={className}
          width="24"
          height="24"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M14 9a2 2 0 0 1-2 2H6l-4 4V4c0-1.1.9-2 2-2h8a2 2 0 0 1 2 2v5Z" />
          <path d="M18 9h2a2 2 0 0 1 2 2v11l-4-4h-6a2 2 0 0 1-2-2v-1" />
        </svg>
      )
      title = 'Chat'
      break
    case ModelHealthBackend.Embedding:
      icon = (
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className={className}
          width="24"
          height="24"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="m9 9-2 2 2 2" />
          <path d="m13 13 2-2-2-2" />
          <circle cx="11" cy="11" r="8" />
          <path d="m21 21-4.3-4.3" />
        </svg>
      )
      title = 'Embedding'
      break
  }

  return (
    <div className="flex items-center gap-1">
      {icon}
      <span className="font-semibold">{title}</span>
    </div>
  )
}

interface CircularProgressProps {
  progress: number
  size?: number
  strokeWidth?: number
  className?: string
  children?: React.ReactNode
  color: string
}

const CircularProgress: React.FC<CircularProgressProps> = ({
  progress,
  size = 40,
  strokeWidth = 4,
  className,
  children,
  color
}) => {
  const radius = (size - strokeWidth) / 2
  const circumference = 2 * Math.PI * radius
  const offset = circumference - (progress / 100) * circumference

  return (
    <div
      className={`relative ${className}`}
      style={{ width: size, height: size }}
    >
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <circle
          stroke="#e6e6e6"
          fill="transparent"
          strokeWidth={strokeWidth}
          r={radius}
          cx={size / 2}
          cy={size / 2}
        />
        <circle
          stroke={color}
          fill="transparent"
          strokeWidth={strokeWidth}
          r={radius}
          cx={size / 2}
          cy={size / 2}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          style={{ transition: 'stroke-dashoffset 0.35s' }}
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center text-sm">
        {children}
      </div>
    </div>
  )
}
