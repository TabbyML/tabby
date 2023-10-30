'use client'

import { useHealth } from '@/lib/hooks/use-health'
import RunnerCard from './components/runner-card'

export default function Runners() {
  const { data: health } = useHealth()

  if (!health) return

  return (
    <div className="flex h-full w-full items-start justify-start">
      <RunnerCard title="Local Runner" health={health} />
    </div>
  )
}
