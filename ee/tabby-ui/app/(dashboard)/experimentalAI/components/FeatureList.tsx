'use client'

import { useEnableCodeBrowserQuickActionBar } from '@/lib/experiment-flags'

import { Switch } from '@/components/ui/switch'

export default function FeatureList () {
  const [quickActionBar, toggleQuickActionBar] = useEnableCodeBrowserQuickActionBar()

  return (
    <>
      {!quickActionBar.loading &&
        <div className="flex items-center space-x-4 rounded-md border p-4">
          <div className="flex-1 space-y-1">
            <p className="text-sm font-medium leading-none">Quick Action Bar</p>
            <p className="text-sm text-muted-foreground">Enable Quick Action Bar to display a convenient toolbar when you select code, offering options to explain the code, add unit tests, and more.</p>
          </div>
          <Switch checked={quickActionBar.value} onCheckedChange={toggleQuickActionBar} />
        </div>
      }
    </>
  )
}