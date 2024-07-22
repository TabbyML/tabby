import dynamic from 'next/dynamic'

import { useEnableAnswerEngineDebugMode } from '@/lib/experiment-flags'
import { useCurrentTheme } from '@/lib/hooks/use-current-theme'
import {
  Drawer,
  DrawerContent,
  DrawerHeader,
  DrawerTitle
} from '@/components/ui/drawer'

const ReactJsonView = dynamic(() => import('react-json-view'), { ssr: false })

interface DebugDrawerProps {
  open: boolean
  onOpenChange: (v: boolean) => void
  value: object | undefined
}

export const DebugDrawer: React.FC<DebugDrawerProps> = ({
  open,
  onOpenChange,
  value
}) => {
  const [enableDebug] = useEnableAnswerEngineDebugMode()
  const { theme } = useCurrentTheme()

  if (!enableDebug?.value) return null

  return (
    <Drawer open={open} onOpenChange={onOpenChange} shouldScaleBackground>
      <DrawerContent className="h-[60vh]">
        <DrawerHeader>
          <DrawerTitle>Debug information</DrawerTitle>
        </DrawerHeader>
        {value ? (
          <div className="overflow-y-auto">
            <ReactJsonView
              theme={theme === 'dark' ? 'tomorrow' : 'rjv-default'}
              src={value}
              style={{ fontSize: '0.75rem' }}
              collapseStringsAfterLength={120}
            />
          </div>
        ) : null}
      </DrawerContent>
    </Drawer>
  )
}
