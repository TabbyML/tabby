import { Worker, WorkerKind } from '@/lib/gql/generates/graphql'
import { cn } from '@/lib/utils'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

type RunnerType = WorkerKind | 'INDEX'

interface RunnerCardProps extends Partial<Omit<Worker, '__typename' | 'kind'>> {
  kind: RunnerType
}

export default function RunnerCard({
  addr,
  name,
  kind,
  device,
  cudaDevices,
  cpuCount,
  cpuInfo
}: RunnerCardProps) {
  const textClass = cn(
    'ml-2',
    'whitespace-nowrap',
    'overflow-hidden',
    'text-ellipsis'
  )
  const cpuMessage = `${cpuInfo} (${cpuCount} cores)`
  return (
    <Card className="rounded-xl p-2 shadow-md lg:w-[260px]">
      <CardHeader className="p-0 px-4 pb-2 pt-4">
        <CardTitle className="text-md flex items-center font-normal">
          <ModelIcon type={kind} />
          <p title={name} className={textClass}>
            {name}
          </p>
        </CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-2 p-0 px-4 pb-4 pt-2">
        <Info>
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className=" h-5 w-5"
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <rect x="16" y="16" width="6" height="6" rx="1" />
            <rect x="2" y="16" width="6" height="6" rx="1" />
            <rect x="9" y="2" width="6" height="6" rx="1" />
            <path d="M5 16v-3a1 1 0 0 1 1-1h12a1 1 0 0 1 1 1v3" />
            <path d="M12 12V8" />
          </svg>
          <p title={addr} className={textClass}>
            {addr}
          </p>
        </Info>
        <Info>
          <svg
            className=" h-5 w-5"
            fill="none"
            height="24"
            stroke="currentColor"
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth="2"
            viewBox="0 0 24 24"
            width="24"
            xmlns="http://www.w3.org/2000/svg"
          >
            <rect height="16" rx="2" width="16" x="4" y="4" />
            <rect height="6" width="6" x="9" y="9" />
            <path d="M15 2v2" />
            <path d="M15 20v2" />
            <path d="M2 15h2" />
            <path d="M2 9h2" />
            <path d="M20 15h2" />
            <path d="M20 9h2" />
            <path d="M9 2v2" />
            <path d="M9 20v2" />
          </svg>
          <p title={cpuMessage} className={textClass}>
            {cpuMessage}
          </p>
        </Info>
        {device == 'cuda' &&
          cudaDevices?.length &&
          cudaDevices.map((x, i) => (
            <Info key={i}>
              <svg
                className=" h-5 w-5"
                fill="none"
                height="24"
                stroke="currentColor"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                viewBox="0 0 24 24"
                width="24"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path d="M6 19v-3" />
                <path d="M10 19v-3" />
                <path d="M14 19v-3" />
                <path d="M18 19v-3" />
                <path d="M8 11V9" />
                <path d="M16 11V9" />
                <path d="M12 11V9" />
                <path d="M2 15h20" />
                <path d="M2 7a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v1.1a2 2 0 0 0 0 3.837V17a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2v-5.1a2 2 0 0 0 0-3.837Z" />
              </svg>
              <p title={x} className={textClass}>
                {x}
              </p>
            </Info>
          ))}
      </CardContent>
    </Card>
  )
}

interface InfoProps {
  children: React.ReactNode
}

function Info({ children }: InfoProps) {
  return (
    <div className="mt-2 flex items-center text-sm text-card-foreground/70">
      {children}
    </div>
  )
}

function ModelIcon({ type }: { type: RunnerType }) {
  const className = 'h-5 w-5'
  if (type == WorkerKind.Completion) {
    return (
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
  } else if (type == WorkerKind.Chat) {
    return (
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
  } else if (type == 'INDEX') {
    return (
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
  }
}
