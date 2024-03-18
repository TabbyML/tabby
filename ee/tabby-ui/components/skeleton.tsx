'use client'

import { Skeleton } from './ui/skeleton'

export const ListSkeleton = () => {
  return (
    <div className="space-y-3">
      <Skeleton className="h-4 w-full" />
      <Skeleton className="h-4 w-full" />
      <Skeleton className="h-4 w-full" />
      <Skeleton className="h-4 w-full" />
    </div>
  )
}

export const FormSkeleton = () => {
  return (
    <div className="flex flex-col gap-3">
      <Skeleton className="h-4 w-[20%]" />
      <Skeleton className="h-4 w-full" />
      <Skeleton className="h-4 w-[20%]" />
      <Skeleton className="h-4 w-full" />
    </div>
  )
}
