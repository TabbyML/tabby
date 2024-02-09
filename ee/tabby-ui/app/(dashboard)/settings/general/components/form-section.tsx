import React from 'react'

import { cn } from '@/lib/utils'

interface GeneralFormSectionProps
  extends Omit<React.HTMLAttributes<HTMLDivElement>, 'title'> {
  title: string
}

export const GeneralFormSection: React.FC<GeneralFormSectionProps> = ({
  title,
  className,
  children,
  ...props
}) => {
  return (
    <div className={cn('lg:flex', className)} {...props}>
      <div className="text-left lg:w-1/5">
        <h1 className="text-2xl font-bold">{title}</h1>
      </div>
      <div className="flex-1 lg:px-4">
        <div className="mb-7 mt-4 lg:mt-0">{children}</div>
      </div>
    </div>
  )
}
