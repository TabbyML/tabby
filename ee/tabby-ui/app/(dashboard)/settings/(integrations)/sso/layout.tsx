import { Metadata } from 'next'

import { cn } from '@/lib/utils'
import { IconExternalLink } from '@/components/ui/icons'

export const metadata: Metadata = {
  title: 'SSO'
}

export default function SSOLayout({ children }: { children: React.ReactNode }) {
  return (
    <>
      <SSOHeader />
      {children}
    </>
  )
}

function SSOHeader({ className }: { className?: string }) {
  return (
    <div className={cn('min-h-8 mb-4 flex items-center gap-4', className)}>
      <div className="flex-1 text-sm text-muted-foreground">
        Single Sign-On (SSO) is an authentication method that enables users to
        authenticate with multiple applications and websites via a single set of
        credentials.
        {false && (
          <a className="ml-2 inline-flex cursor-pointer flex-row items-center text-primary hover:underline">
            Learn more
            <IconExternalLink />
          </a>
        )}
      </div>
    </div>
  )
}
