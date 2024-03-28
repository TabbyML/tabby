'use client'

import { useEnableCodeBrowserQuickActionBar } from '@/lib/experiment-flags'
import { Switch } from '@/components/ui/switch'

export default function FeatureList() {
  const [quickActionBar, toggleQuickActionBar] =
    useEnableCodeBrowserQuickActionBar()
  return (
    <>
      {!quickActionBar.loading && (
        <div className="flex items-center space-x-4 rounded-md border p-4">
          <div className="flex-1 space-y-1">
            <p className="text-sm font-medium leading-none">
              {quickActionBar.title}
            </p>
            <p className="text-sm text-muted-foreground">
              {quickActionBar.description}
            </p>
          </div>
          <Switch
            checked={quickActionBar.value}
            onCheckedChange={toggleQuickActionBar}
          />
        </div>
      )}
    </>
  )
}
