import { cn } from '@/lib/utils'
import { IconExternalLink } from '@/components/ui/icons'

export const RepositoryHeader = ({
  extra,
  className
}: {
  extra?: React.ReactNode
  className?: string
}) => {
  return (
    <div className={cn('min-h-8 mb-4 flex items-center gap-4', className)}>
      <div className="flex-1 text-sm text-muted-foreground">
        GitHub reposotiry.
        {false && (
          <a className="ml-2 inline-flex cursor-pointer flex-row items-center text-primary hover:underline">
            Learn more
            <IconExternalLink />
          </a>
        )}
      </div>
      <div className="h-8 shrink-0 self-start">{extra}</div>
    </div>
  )
}
