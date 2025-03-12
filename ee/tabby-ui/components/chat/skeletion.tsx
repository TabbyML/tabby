import { Skeleton } from '../ui/skeleton'

export function QaPairSkeleton() {
  return (
    <div className="space-y-8">
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <Skeleton className="h-8 w-8 rounded-full" />
        </div>
        <div className="space-y-3">
          <Skeleton className="w-full" />
          <Skeleton className="w-[60%]" />
        </div>
      </div>
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <Skeleton className="h-8 w-8 rounded-full" />
        </div>
        <div className="space-y-3">
          <Skeleton className="w-full" />
          <Skeleton className="w-full" />
          <Skeleton className="w-[80%]" />
        </div>
      </div>
    </div>
  )
}
