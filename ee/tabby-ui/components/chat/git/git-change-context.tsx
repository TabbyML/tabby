import React from 'react'
import { AnimatePresence, motion } from 'framer-motion'

import { cn } from '@/lib/utils'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { IconFileText, IconRemove } from '@/components/ui/icons'

export interface GitChange {
  id: string
  filepath: string
  additions: number
  deletions: number
  diffContent: string
  lineStart?: number
}

interface GitChangesContextProps {
  changes: GitChange[]
  onRemoveChange: (id: string) => void
  onClick?: (filepath: string) => void
  className?: string
}

export function GitChangesContext({
  changes,
  onRemoveChange,
  onClick,
  className
}: GitChangesContextProps) {
  if (!changes.length) return null

  return (
    <AnimatePresence>
      {changes.map(change => (
        <motion.div
          key={change.id}
          initial={{ opacity: 0, scale: 0.9, y: -5 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          transition={{
            ease: 'easeInOut',
            duration: 0.1
          }}
          exit={{ opacity: 0, scale: 0.9, y: -5 }}
          layout
          className="overflow-hidden"
        >
          <Badge
            variant="outline"
            className={cn(
              'inline-flex h-7 w-full cursor-pointer flex-nowrap items-center gap-1 overflow-hidden rounded-md pr-0 text-sm font-semibold',
              className
            )}
            onClick={() => onClick?.(change.filepath)}
          >
            <IconFileText className="shrink-0" />
            <span className="truncate">{change.filepath.split('/').pop()}</span>
            <span className="ml-1">
              <span className="text-green-500">+{change.additions}</span>{' '}
              <span className="text-red-500">-{change.deletions}</span>
            </span>
            <Button
              size="icon"
              variant="ghost"
              className="h-7 w-7 shrink-0 rounded-l-none hover:bg-muted/50"
              onClick={e => {
                e.stopPropagation()
                onRemoveChange(change.id)
              }}
            >
              <IconRemove />
            </Button>
          </Badge>
        </motion.div>
      ))}
    </AnimatePresence>
  )
}
