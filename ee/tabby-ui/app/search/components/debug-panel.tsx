import React, { lazy, Suspense, useState } from 'react'

import { useEnableAnswerEngineDebugMode } from '@/lib/experiment-flags'
import { useCurrentTheme } from '@/lib/hooks/use-current-theme'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { IconChevronDown, IconClose } from '@/components/ui/icons'
import { ScrollArea } from '@/components/ui/scroll-area'
import { ListSkeleton } from '@/components/skeleton'

const ReactJsonView = lazy(() => import('react-json-view'))

interface DebugDrawerProps {
  open: boolean
  onOpenChange: (v: boolean) => void
  value: object | undefined
}

export const DebugPanel: React.FC<DebugDrawerProps> = ({
  open,
  onOpenChange,
  value
}) => {
  const [enableDebug] = useEnableAnswerEngineDebugMode()
  const { theme } = useCurrentTheme()
  const [fullScreen, setFullScreen] = useState(false)

  if (!enableDebug?.value) return null

  if (!open) return null

  return (
    <div className="flex h-full flex-col px-3 pt-2">
      <div className="flex items-center justify-between pb-2">
        <span className="font-semibold">Debug</span>
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="icon">
            <IconChevronDown
              className={cn('transition-all', fullScreen ? '' : 'rotate-180')}
            />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => onOpenChange(false)}
          >
            <IconClose />
          </Button>
        </div>
      </div>
      <Suspense fallback={<ListSkeleton className="p-2" />}>
        {value ? (
          <ScrollArea className="flex-1">
            <ReactJsonView
              theme={theme === 'dark' ? 'tomorrow' : 'rjv-default'}
              src={value}
              style={{ fontSize: '0.75rem' }}
              collapseStringsAfterLength={120}
            />
          </ScrollArea>
        ) : null}
      </Suspense>
    </div>
  )
}
