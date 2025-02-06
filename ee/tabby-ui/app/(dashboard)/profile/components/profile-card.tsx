import React from 'react'

import { useMe } from '@/lib/hooks/use-me'
import { cn } from '@/lib/utils'
import { CardContent, CardTitle } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'

interface ProfileCardProps extends React.HTMLAttributes<HTMLDivElement> {
  title: string
  description?: string
  footer?: React.ReactNode
  footerClassname?: string
  hideForSsoUser?: boolean
}

const ProfileCard: React.FC<ProfileCardProps> = ({
  title,
  description,
  footer,
  footerClassname,
  className,
  hideForSsoUser,
  children,
  ...props
}) => {
  const [{ data }] = useMe()
  const isSsoUser = data?.me?.isSsoUser

  if (isSsoUser && hideForSsoUser) {
    return null
  }

  return (
    <div
      className={cn(
        'flex w-full flex-col gap-8 rounded-lg border p-6 pb-0 xl:w-[800px]',
        className
      )}
      {...props}
    >
      <div>
        <CardTitle>{title}</CardTitle>
        {description && (
          <div className="mt-4 text-sm text-muted-foreground">
            {description}
          </div>
        )}
      </div>
      <CardContent className="p-0">{children}</CardContent>
      <div
        className={cn(
          'rounded-b-lg pb-6 text-sm text-muted-foreground',
          footerClassname
        )}
      >
        {!!footer && <Separator className="mb-6" />}
        {footer}
      </div>
    </div>
  )
}

export { ProfileCard }
