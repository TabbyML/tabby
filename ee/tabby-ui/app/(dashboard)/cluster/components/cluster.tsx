'use client'

import { noop } from 'lodash-es'
import { useQuery } from 'urql'
import bytes from 'bytes';
import { useTheme } from 'next-themes';

import { graphql } from '@/lib/gql/generates'
import { WorkerKind } from '@/lib/gql/generates/graphql'
import { useHealth } from '@/lib/hooks/use-health'
import { useWorkers } from '@/lib/hooks/use-workers'
import { useMutation } from '@/lib/tabby/gql'
import { Button } from '@/components/ui/button'
import { IconRotate } from '@/components/ui/icons'
import { Input } from '@/components/ui/input'
import { Separator } from '@/components/ui/separator'
import { CopyButton } from '@/components/copy-button'
import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Label,
} from 'recharts'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { IconFolderOpen } from '@/components/ui/icons';

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

  // TODO: mock data
  const usageData = [
    { name: "Models", usage: 1024 * 1024 * 1024 * 1.1, color: "#0088FE", folders: ["~/.tabby/models"] },
    { name: "Repository data", usage: 1024 * 1024 * 2.1, color: "#00C49F", folders: ["~/.tabby/dataset", "~/.tabby/index"] },
    { name: "Database", usage: 1024 * 85, color: "#FFBB28", folders: ["~/.tabby/ee"] },
    { name: "Events", usage: 1024 * 1200 * 11, color: "#FF8042", folders: ["~/.tabby/events"] }
  ];
  const totalUsage = 1024 * 1300 * 1200

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

      <Separator className="mt-5" />

      <div>
        <p className="font-bold">
          Storage Usage
        </p>
        <p className="text-sm text-muted-foreground">Disk space used by Tabby</p>
      </div>
      
      
      <div className="flex flex-col items-center gap-x-3 md:flex-row">
        <ResponsiveContainer width={230} height={220}>
          <PieChart>
            <Pie 
              data={usageData} 
              dataKey="usage"
              cx={110}
              cy={100}
              innerRadius={70}
              outerRadius={90}
              stroke='none'
            >
              {usageData.map((entry) => (
                <Cell key={entry.name} fill={entry.color} />
              ))}
              <Label  
                content={<CustomLabel totalUsage={totalUsage} />}
                position="center"
              />
            </Pie>
          </PieChart>
        </ResponsiveContainer>

        <div className="flex w-full flex-col gap-y-2 md:ml-10 md:w-auto">
          {usageData.map(item => (
            <Tooltip key={item.name}>
              <TooltipTrigger>
                <div className="flex cursor-default items-center justify-between text-xs" >
                  <div className="flex w-40 items-center">
                    <div className="mr-1.5 h-3 w-3 rounded" style={{ backgroundColor: item.color }} />
                    <p className="font-semibold">{item.name}</p>
                  </div>
                  <p>{bytes(item.usage)}</p>
                </div>
              </TooltipTrigger>
              <TooltipContent>
                <div className="flex items-center text-muted-foreground">
                  <IconFolderOpen className="mr-0.5" />
                  <p>Folder</p>
                </div>
                
                {item.folders.map(folder => (<p>{folder}</p>))}
              </TooltipContent>
            </Tooltip>
          ))}
        </div>
      </div>
    </div>
  )
}

function CustomLabel ({
  viewBox,
  totalUsage
}: {
  viewBox?: {
    cx: number;
    cy: number;
  };
  totalUsage: number;
}) {
  const { theme } = useTheme()
  if (!viewBox) return
  const { cx, cy } = viewBox
  return (
    <g>
      <text
        x={cx}
        y={cy - 10}
        textAnchor="middle"
        dominantBaseline="central"
        alignmentBaseline="middle"
        fill={theme === 'dark' ? '#FDFDFD' : '#030302'}
        className="text-sm"
      >
        Total Usuage
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
        {bytes(totalUsage)}
      </text>
    </g>
  );
};
