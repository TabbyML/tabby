'use client'

import { Switch } from '@/components/ui/switch'

export default function ExperimentalAIOption({
  title,
  description
}: {
  title: string
  description: string
}) {
  return (
    <div className="flex items-center space-x-4 rounded-md border p-4">
      <div className="flex-1 space-y-1">
        <p className="text-sm font-medium leading-none ">{title}</p>
        <p className="text-sm text-muted-foreground">{description}</p>
      </div>
      <Switch />
    </div>
  )
}
