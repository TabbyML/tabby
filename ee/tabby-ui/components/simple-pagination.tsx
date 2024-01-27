import React from 'react'

import { cn } from '@/lib/utils'

import { Button } from './ui/button'
import { IconChevronRight } from './ui/icons'

interface SimplePagination extends React.HTMLAttributes<HTMLDivElement> {
  hasPreviousPage: boolean | undefined
  hasNextPage: boolean | undefined
  onNext: () => void
  onPrev: () => void
}
const SimplePagination: React.FC<SimplePagination> = ({
  className,
  hasPreviousPage,
  hasNextPage,
  onNext,
  onPrev
}) => {
  return (
    <div className={cn('flex items-center gap-2', className)}>
      <Button disabled={!hasPreviousPage} onClick={onPrev}>
        <IconChevronRight className="rotate-180" />{' '}
      </Button>
      <Button disabled={!hasNextPage} onClick={onNext}>
        <IconChevronRight />{' '}
      </Button>
    </div>
  )
}

export { SimplePagination }
