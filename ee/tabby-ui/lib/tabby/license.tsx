import * as React from 'react'
import Link from 'next/link'
import { Slot } from '@radix-ui/react-slot'
import { capitalize } from 'lodash-es'
import { useQuery } from 'urql'

import { buttonVariants } from '@/components/ui/button'
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger
} from '@/components/ui/hover-card'

import {
  GetLicenseInfoQuery,
  LicenseStatus,
  LicenseType
} from '../gql/generates/graphql'
import { getLicenseInfo } from './query'

interface LicenseProviderProps {
  children: React.ReactNode
}

interface LicenseContextValue {
  license: GetLicenseInfoQuery['license'] | undefined | null
  refreshLicense: () => void
}

const LicenseContext = React.createContext<LicenseContextValue>(
  {} as LicenseContextValue
)

const LicenseProvider: React.FunctionComponent<LicenseProviderProps> = ({
  children
}) => {
  const [{ data }, refreshLicense] = useQuery({ query: getLicenseInfo })

  return (
    <LicenseContext.Provider value={{ license: data?.license, refreshLicense }}>
      {children}
    </LicenseContext.Provider>
  )
}

class AuthProviderIsMissing extends Error {
  constructor() {
    super(
      'LicenseProvider is missing. Please add the LicenseProvider at root level'
    )
  }
}

function useLicense() {
  const context = React.useContext(LicenseContext)

  if (!context) {
    throw new AuthProviderIsMissing()
  }

  return context
}

interface LicenseGuardProps
  extends React.ComponentPropsWithoutRef<typeof Slot> {
  licenses?: LicenseType[]
}

const LicenseGuard = React.forwardRef<
  React.ElementRef<typeof Slot>,
  LicenseGuardProps
>(({ licenses, children }, ref) => {
  const [open, setOpen] = React.useState(false)
  const { license } = useLicense()
  let isLicenseDisabled = false
  if (licenses?.length) {
    if (
      !license ||
      license?.status !== LicenseStatus.Ok ||
      !licenses.includes(license?.type)
    ) {
      isLicenseDisabled = true
    }
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

  const licensesString = licenses?.map(l => capitalize(l))?.join('/') ?? ''

  return (
    <HoverCard open={open} onOpenChange={onOpenChange} openDelay={100}>
      <HoverCardContent side="top" collisionPadding={16} className="w-[400px]">
        <div>
          This feature is only available on Tabby’s {licensesString} license.
          Upgrade to use this feature.
        </div>
        <div className="text-center mt-4">
          <Link className={buttonVariants()} href="/settings/subscription">
            Upgrade to {licensesString}
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
