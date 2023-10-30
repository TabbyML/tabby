import { CardTitle, CardHeader, CardContent, Card } from '@/components/ui/card'
import { HealthInfo } from '@/lib/hooks/use-health'

interface RunnerCardProps {
  title: string
  health: HealthInfo
}

export default function RunnerCard({ title, health }: RunnerCardProps) {
  return (
    <Card className="rounded-xl shadow-md">
      <CardHeader className="p-0 px-4 pb-2 pt-4">
        <CardTitle className="text-md font-semibold">{title}</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-2 p-0 px-4 pb-4 pt-2">
        <Info>
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className=" h-5 w-5 text-gray-400"
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
          <p className="ml-2">{health.model}</p>
        </Info>
        {health.chat_model && (
          <Info>
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className=" h-5 w-5 text-gray-400"
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
            <p className="ml-2">{health.chat_model}</p>
          </Info>
        )}
        <Info>
          <svg
            className=" h-5 w-5 text-gray-400"
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
          <p className="ml-2">
            {health.cpu_info} ({health.cpu_count} cores)
          </p>
        </Info>
        {health.cuda_devices.map((x, i) => (
          <Info key={i}>
            <svg
              className=" h-5 w-5 text-gray-400"
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
            <p className="ml-2">{x}</p>
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
    <div className="mt-2 flex items-center text-sm text-gray-500">
      {children}
    </div>
  )
}
