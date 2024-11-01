import * as React from 'react'
import Link from 'next/link'
import { capitalize } from 'lodash-es'

import {
  GetLicenseInfoQuery,
  LicenseInfo,
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
  isSeatCountRelated?: boolean
  children: (params: {
    hasValidLicense: boolean
    license: GetLicenseInfoQuery['license'] | undefined | null
  }) => React.ReactNode
}

const LicenseGuard: React.FC<LicenseGuardProps> = ({
  licenses,
  isSeatCountRelated,
  children
}) => {
  const [open, setOpen] = React.useState(false)
  const license = useLicenseInfo()

  const hasValidLicense =
    !!license &&
    licenses.includes(license.type) &&
    license.status === LicenseStatus.Ok

  const onOpenChange = (v: boolean) => {
    if (hasValidLicense) return
    setOpen(v)
  }

  let licenseString = capitalize(licenses[0])
  let licenseText = licenseString
  if (licenses.length == 2) {
    licenseText = `${capitalize(licenses[0])} or ${capitalize(licenses[1])}`
  }

  return (
    <HoverCard open={open} onOpenChange={onOpenChange} openDelay={100}>
      <HoverCardContent side="top" collisionPadding={16} className="w-[400px]">
        <LicenseTips
          licenses={licenses}
          license={license}
          isSeatCountRelated={!!isSeatCountRelated}
        />
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
  licenses,
  license,
  isSeatCountRelated
}: {
  licenses: LicenseType[]
  license: LicenseInfo | undefined
  isSeatCountRelated: boolean
}) {
  const hasSufficientLicense = !!license && licenses.includes(license.type)
  const isLicenseExpired =
    hasSufficientLicense && license?.status === LicenseStatus.Expired
  const isSeatsExceeded =
    hasSufficientLicense && license?.status === LicenseStatus.SeatsExceeded

  const licenseString = capitalize(licenses[0])
  let insufficientLicenseText = licenseString
  if (licenses.length == 2) {
    insufficientLicenseText = `${capitalize(licenses[0])} or ${capitalize(
      licenses[1]
    )}`
  }

  if (isSeatsExceeded && isSeatCountRelated) {
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
        ``
      </>
    )
  }

  if (isLicenseExpired) {
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
