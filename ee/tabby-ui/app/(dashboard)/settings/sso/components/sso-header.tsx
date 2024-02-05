import { cn } from '@/lib/utils'
import { IconExternalLink } from '@/components/ui/icons'

export const SSOHeader = ({
  extra,
  className
}: {
  extra?: React.ReactNode
  className?: string
}) => {
  return (
    <div className={cn('mb-4 flex items-center gap-4 min-h-8', className)}>
      <div className="flex-1 text-sm text-muted-foreground">
        Single Sign-On (SSO) is an authentication method that enables users to
        authenticate with multiple applications and websites via a single set of
        credentials.
        {false && <a className="ml-2 inline-flex cursor-pointer flex-row items-center text-primary hover:underline">
          Learn more
          <IconExternalLink />
        </a>}
      </div>
      <div className="shrink-0 h-8">{extra}</div>
    </div>
  )
}
