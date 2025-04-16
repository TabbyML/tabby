import React, { lazy, Suspense, useEffect, useRef } from 'react'

import { useEnableDeveloperMode } from '@/lib/experiment-flags'
import { useCopyToClipboard } from '@/lib/hooks/use-copy-to-clipboard'
import { useCurrentTheme } from '@/lib/hooks/use-current-theme'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { IconChevronDown, IconClose } from '@/components/ui/icons'
import { ScrollArea } from '@/components/ui/scroll-area'
import { ListSkeleton } from '@/components/skeleton'

const ReactJsonView = lazy(() => import('react-json-view'))

interface DevPanelProps {
  isFullScreen: boolean
  onToggleFullScreen: (fullScreen: boolean) => void
  value: object | undefined
  onClose: () => void
  scrollOnUpdate?: boolean
}

export const DevPanel: React.FC<DevPanelProps> = ({
  value,
  isFullScreen,
  onToggleFullScreen,
  onClose,
  scrollOnUpdate = true
}) => {
  const [enableDeveloperMode] = useEnableDeveloperMode()
  const { theme } = useCurrentTheme()
  const scrollAreaRef = useRef<HTMLDivElement>(null)
  const { copyToClipboard } = useCopyToClipboard({ timeout: 0 })

  useEffect(() => {
    if (value && scrollOnUpdate) {
      scrollAreaRef.current?.scrollTo({
        top: 0,
        behavior: 'smooth'
      })
    }
  }, [value])

  if (!enableDeveloperMode?.value) return null

  if (!open) return null

  return (
    <div className="flex h-full flex-col px-3 pt-2">
      <div className="flex items-center justify-end pb-2">
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="icon"
            onClick={e => onToggleFullScreen(!isFullScreen)}
          >
            <IconChevronDown
              className={cn('transition-all', isFullScreen ? '' : 'rotate-180')}
            />
          </Button>
          <Button variant="ghost" size="icon" onClick={onClose}>
            <IconClose />
          </Button>
        </div>
      </div>
      <Suspense fallback={<ListSkeleton className="p-2" />}>
        {value ? (
          <ScrollArea className="flex-1" ref={scrollAreaRef}>
            <ReactJsonView
              theme={theme === 'dark' ? 'tomorrow' : 'rjv-default'}
              src={value}
              style={{ fontSize: '0.75rem' }}
              collapseStringsAfterLength={120}
              displayObjectSize={false}
              displayDataTypes={false}
              enableClipboard={({ src }) => {
                if (typeof src === 'string') {
                  copyToClipboard(src)
                }
              }}
            />
          </ScrollArea>
        ) : null}
      </Suspense>
    </div>
  )
}
