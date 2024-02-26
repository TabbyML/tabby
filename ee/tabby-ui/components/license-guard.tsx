import * as React from 'react'
import Link from 'next/link'
import { capitalize } from 'lodash-es'

import {
  GetLicenseInfoQuery,
  LicenseStatus,
  LicenseType
} from '@/lib/gql/generates/graphql'
import { useLicenseInfo } from '@/lib/hooks/use-license'
import { cn } from '@/lib/utils'
import { buttonVariants } from '@/components/ui/button'
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger
} from '@/components/ui/hover-card'

interface LicenseGuardProps {
  licenses: LicenseType[]
  children:
    | React.ReactNode
    | React.ReactNode[]
    | ((params: {
        disabled: boolean
        license: GetLicenseInfoQuery['license'] | undefined | null
      }) => React.ReactNode)
}

const LicenseGuard: React.FC<LicenseGuardProps> = ({ licenses, children }) => {
  const [open, setOpen] = React.useState(false)
  const license = useLicenseInfo()
  let isLicenseDisabled = false
  if (
    !license ||
    license?.status !== LicenseStatus.Ok ||
    !licenses.includes(license?.type)
  ) {
    isLicenseDisabled = true
  }

  const updatedChildren = React.useMemo(() => {
    if (typeof children === 'function') {
      return null
    }
    return React.Children.map(children, child => {
      if (React.isValidElement(child)) {
        return React.cloneElement(child, {
          disabled: isLicenseDisabled || child.props.disabled
        } as React.Attributes)
      }
      return child
    })
  }, [children, isLicenseDisabled])

  const onOpenChange = (v: boolean) => {
    if (!isLicenseDisabled) return
    setOpen(v)
  }

  let licenseString = capitalize(licenses[0])
  let licenseText = licenseString
  if (licenses.length > 1) {
    licenseText = `${licenseString} or higher`
  }

  return (
    <HoverCard open={open} onOpenChange={onOpenChange} openDelay={100}>
      <HoverCardContent side="top" collisionPadding={16} className="w-[400px]">
        <div>
          This feature is only available on Tabby’s{' '}
          <span className="font-semibold">{licenseText}</span> plan. Upgrade to
          use this feature.
        </div>
        <div className="mt-4 text-center">
          <Link className={buttonVariants()} href="/settings/subscription">
            Upgrade to {licenseText}
          </Link>
        </div>
      </HoverCardContent>
      <HoverCardTrigger
        asChild
        onClick={e => {
          e.preventDefault()
          onOpenChange(true)
        }}
      >
        <div className={cn(isLicenseDisabled ? 'cursor-not-allowed' : '')}>
          {typeof children === 'function'
            ? children({ disabled: isLicenseDisabled, license })
            : updatedChildren}
        </div>
      </HoverCardTrigger>
    </HoverCard>
  )
}
LicenseGuard.displayName = 'LicenseGuard'

export { LicenseGuard }
