import * as React from 'react'
import Link from 'next/link'
import { capitalize } from 'lodash-es'

import { GetLicenseInfoQuery, LicenseType } from '@/lib/gql/generates/graphql'
import {
  useLicenseInfo,
  useLicenseValidity,
  UseLicenseValidityResponse
} from '@/lib/hooks/use-license'
import { cn } from '@/lib/utils'
import { buttonVariants } from '@/components/ui/button'
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger
} from '@/components/ui/hover-card'

interface LicenseGuardProps {
  /**
   * requiredLicenses
   */
  licenses: LicenseType[]
  children: (params: {
    hasValidLicense: boolean
    license: GetLicenseInfoQuery['license'] | undefined | null
  }) => React.ReactNode
}

const LicenseGuard: React.FC<LicenseGuardProps> = ({ licenses, children }) => {
  const [open, setOpen] = React.useState(false)
  const license = useLicenseInfo()

  const licenseValidity = useLicenseValidity({ licenses })
  const { isLicenseOK, hasSufficientLicense } = licenseValidity

  const hasValidLicense = hasSufficientLicense && isLicenseOK

  const onOpenChange = (v: boolean) => {
    if (hasValidLicense) return
    setOpen(v)
  }

  return (
    <HoverCard open={open} onOpenChange={onOpenChange} openDelay={100}>
      <HoverCardContent side="top" collisionPadding={16} className="w-[400px]">
        <LicenseTips licenses={licenses} {...licenseValidity} />
      </HoverCardContent>
      <HoverCardTrigger
        asChild
        onClick={e => {
          if (!hasValidLicense) {
            e.preventDefault()
            onOpenChange(true)
          }
        }}
      >
        <div className={cn(!hasValidLicense ? 'cursor-not-allowed' : '')}>
          {children({ hasValidLicense, license })}
        </div>
      </HoverCardTrigger>
    </HoverCard>
  )
}
LicenseGuard.displayName = 'LicenseGuard'

export { LicenseGuard }

function LicenseTips({
  hasSufficientLicense,
  isExpired,
  isSeatsExceeded,
  licenses
}: UseLicenseValidityResponse & {
  licenses: LicenseType[]
}) {
  const licenseString = capitalize(licenses[0])
  let insufficientLicenseText = licenseString
  if (licenses.length == 2) {
    insufficientLicenseText = `${capitalize(licenses[0])} or ${capitalize(
      licenses[1]
    )}`
  }

  // for expired sufficient license
  if (hasSufficientLicense && isExpired) {
    return (
      <>
        <div>
          Your license has expired. Please update your license to use this
          feature.
        </div>
        <div className="mt-4 text-center">
          <Link className={buttonVariants()} href="/settings/subscription">
            Update license
          </Link>
        </div>
      </>
    )
  }

  // for seatsExceeded sufficient license
  if (hasSufficientLicense && isSeatsExceeded) {
    return (
      <>
        <div>
          Your seat count has exceeded the limit. Please upgrade your license to
          continue using this feature.
        </div>
        <div className="mt-4 text-center">
          <Link className={buttonVariants()} href="/settings/subscription">
            Upgrade license
          </Link>
        </div>
      </>
    )
  }

  return (
    <>
      <div>
        This feature is only available on Tabby&apos;s{' '}
        <span className="font-semibold">{insufficientLicenseText}</span> plan.
        Upgrade to use this feature.
      </div>
      <div className="mt-4 text-center">
        <Link className={buttonVariants()} href="/settings/subscription">
          Upgrade to {licenseString}
        </Link>
      </div>
    </>
  )
}
