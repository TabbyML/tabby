import { Skeleton } from '@/components/ui/skeleton'

export function MessagesSkeleton() {
  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <Skeleton className="w-full" />
        <Skeleton className="w-[70%]" />
      </div>
      <Skeleton className="h-40 w-full" />
    </div>
  )
}
