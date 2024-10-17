import { cn } from '@/lib/utils'

function Skeleton({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn('h-4 animate-pulse rounded-md bg-border', className)}
      {...props}
    />
  )
}

export { Skeleton }
