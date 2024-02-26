import * as React from 'react'
import Link from 'next/link'
import { Slot } from '@radix-ui/react-slot'
import { capitalize } from 'lodash-es'
import { useQuery, UseQueryState } from 'urql'

import { graphql } from '@/lib/gql/generates'
import {
  Exact,
  GetLicenseInfoQuery,
  LicenseStatus,
  LicenseType
} from '@/lib/gql/generates/graphql'
import { buttonVariants } from '@/components/ui/button'
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger
} from '@/components/ui/hover-card'

export const getLicenseInfo = graphql(/* GraphQL */ `
  query GetLicenseInfo {
    license {
      type
      status
      seats
      seatsUsed
      issuedAt
      expiresAt
    }
  }
`)

interface LicenseProviderProps {
  children: React.ReactNode
}

interface LicenseContextValue {
  licenseInfoQuery: UseQueryState<
    GetLicenseInfoQuery,
    Exact<{
      [key: string]: never
    }>
  >
  license: GetLicenseInfoQuery['license'] | undefined | null
  refreshLicense: () => void
}

const LicenseContext = React.createContext<LicenseContextValue>(
  {} as LicenseContextValue
)

const LicenseProvider: React.FunctionComponent<LicenseProviderProps> = ({
  children
}) => {
  const [licenseInfoQuery, refreshLicense] = useQuery({ query: getLicenseInfo })
  const license = licenseInfoQuery?.data?.license

  return (
    <LicenseContext.Provider
      value={{ licenseInfoQuery, license, refreshLicense }}
    >
      {children}
    </LicenseContext.Provider>
  )
}

class LicenseProviderIsMissing extends Error {
  constructor() {
    super(
      'LicenseProvider is missing. Please add the LicenseProvider at root level'
    )
  }
}

function useLicense() {
  const context = React.useContext(LicenseContext)

  if (!context) {
    throw new LicenseProviderIsMissing()
  }

  return context
}

interface LicenseGuardProps
  extends React.ComponentPropsWithoutRef<typeof Slot> {
  licenses: LicenseType[]
}

const LicenseGuard = React.forwardRef<
  React.ElementRef<typeof Slot>,
  LicenseGuardProps
>(({ licenses, children }, ref) => {
  const [open, setOpen] = React.useState(false)
  const { license } = useLicense()
  let isLicenseDisabled = false
  if (
    !license ||
    license?.status !== LicenseStatus.Ok ||
    !licenses.includes(license?.type)
  ) {
    isLicenseDisabled = true
  }

  const updatedChildren = React.Children.map(children, child => {
    if (React.isValidElement(child)) {
      return React.cloneElement(child, {
        disabled: isLicenseDisabled || child.props.disabled
      } as React.Attributes)
    }
    return child
  })

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
        <div className={isLicenseDisabled ? 'cursor-not-allowed' : ''}>
          {updatedChildren}
        </div>
      </HoverCardTrigger>
    </HoverCard>
  )
})
LicenseGuard.displayName = 'LicenseGuard'

export { LicenseProvider, LicenseGuard, useLicense }
