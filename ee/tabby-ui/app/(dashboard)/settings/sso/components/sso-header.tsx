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
    <div className={cn('mb-4 flex items-center gap-4', className)}>
      <div className="text-muted-foreground text-sm flex-1">
        Single Sign-On (SSO) is an authentication method that enables users to
        authenticate with multiple applications and websites via a single set of
        credentials.
        <a className="ml-2 cursor-pointer text-primary hover:underline inline-flex flex-row items-center">
          Learn more
          <IconExternalLink />
        </a>
      </div>
      <div className="shrink-0">{extra}</div>
    </div>
  )
}
