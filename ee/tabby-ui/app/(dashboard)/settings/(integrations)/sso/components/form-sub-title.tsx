import { cn } from '@/lib/utils'

export function SubTitle({
  className,
  ...rest
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn('mt-4 text-xl font-semibold', className)} {...rest} />
  )
}
