'use client'

import { useEnableDeveloperMode } from '@/lib/experiment-flags'
import { Switch } from '@/components/ui/switch'

export default function FeatureList() {
  const [developerMode, toggleDeveloperMode] = useEnableDeveloperMode()
  return (
    <>
      {!developerMode.loading && (
        <div className="flex items-center space-x-4 rounded-md border p-4">
          <div className="flex-1 space-y-1">
            <p className="text-sm font-medium leading-none">
              {developerMode.title}
            </p>
            <p className="text-sm text-muted-foreground">
              {developerMode.description}
            </p>
          </div>
          <Switch
            checked={developerMode.value}
            onCheckedChange={toggleDeveloperMode}
          />
        </div>
      )}
    </>
  )
}
