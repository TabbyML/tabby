import { cn } from '@/lib/utils'

export function SubTitle({
  className,
  ...rest
}: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn('text-xl font-semibold', className)} {...rest} />
}
