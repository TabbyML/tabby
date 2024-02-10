import React from 'react'
import Link from 'next/link'

import { cn } from '@/lib/utils'
import { IconExternalLink } from '@/components/ui/icons'

interface SubHeaderProps extends React.HTMLAttributes<HTMLDivElement> {
  externalLink?: string
}

export const SubHeader: React.FC<SubHeaderProps> = ({
  className,
  externalLink,
  children
}) => {
  return (
    <div className={cn('mb-4 flex items-center gap-4', className)}>
      <div className="flex-1 text-sm text-muted-foreground">
        {children}
        {!!externalLink && (
          <Link
            className="ml-2 inline-flex cursor-pointer flex-row items-center text-primary hover:underline"
            href={externalLink}
            target="_blank"
          >
            Learn more
            <IconExternalLink />
          </Link>
        )}
      </div>
    </div>
  )
}
