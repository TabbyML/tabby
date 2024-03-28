'use client'

import { HTMLAttributes } from 'react'

import { cn } from '@/lib/utils'

import { Skeleton } from './ui/skeleton'

export const ListSkeleton: React.FC<HTMLAttributes<HTMLDivElement>> = ({
  className,
  ...props
}) => {
  return (
    <div className={cn('space-y-3', className)} {...props}>
      <Skeleton className="h-4 w-full" />
      <Skeleton className="h-4 w-full" />
      <Skeleton className="h-4 w-full" />
      <Skeleton className="h-4 w-full" />
    </div>
  )
}

export const ListRowSkeleton: React.FC<HTMLAttributes<HTMLDivElement>> = ({
  className,
  ...props
}) => {
  return <Skeleton className={cn('h-4 w-full', className)} {...props} />
}

export const FormSkeleton: React.FC<HTMLAttributes<HTMLDivElement>> = ({
  className,
  ...props
}) => {
  return (
    <div className={cn('flex flex-col gap-3', className)} {...props}>
      <Skeleton className="h-4 w-[20%]" />
      <Skeleton className="h-4 w-full" />
      <Skeleton className="h-4 w-[20%]" />
      <Skeleton className="h-4 w-full" />
    </div>
  )
}
