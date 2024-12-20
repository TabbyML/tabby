'use client'

import React, { useMemo } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'

import { useLicenseValidity } from '@/lib/hooks/use-license'
import { cn } from '@/lib/utils'
import { IconClose, IconNotice } from '@/components/ui/icons'

import { buttonVariants } from './ui/button'

export const BANNER_HEIGHT = '3.5rem'

interface ShowLicenseBannerContextValue {
  isShowLicenseBanner: boolean
  setIsShowLicenseBanner: React.Dispatch<React.SetStateAction<boolean>>
}

const ShowLicenseBannerContext =
  React.createContext<ShowLicenseBannerContextValue>(
    {} as ShowLicenseBannerContextValue
  )

export const ShowLicenseBannerProvider = ({
  children
}: {
  children: React.ReactNode
}) => {
  const { isExpired, isSeatsExceeded, isLicenseOK } = useLicenseValidity()

  const [isShowLicenseBanner, setIsShowLicenseBanner] = React.useState(false)

  React.useEffect(() => {
    const isInIframe = window.self !== window.top
    if (isInIframe) return

    if (isExpired || isSeatsExceeded) {
      setIsShowLicenseBanner(true)
    } else if (isLicenseOK) {
      setIsShowLicenseBanner(false)
    }
  }, [isLicenseOK, isExpired, isSeatsExceeded])

  return (
    <ShowLicenseBannerContext.Provider
      value={{ isShowLicenseBanner, setIsShowLicenseBanner }}
    >
      {children}
    </ShowLicenseBannerContext.Provider>
  )
}

export function useShowLicenseBanner(): [
  boolean,
  React.Dispatch<React.SetStateAction<boolean>>
] {
  const { isShowLicenseBanner, setIsShowLicenseBanner } = React.useContext(
    ShowLicenseBannerContext
  )
  return [isShowLicenseBanner, setIsShowLicenseBanner]
}

export function LicenseBanner() {
  const [isShowLicenseBanner, setIsShowLicenseBanner] = useShowLicenseBanner()
  const { isExpired, isSeatsExceeded } = useLicenseValidity()
  const pathname = usePathname()
  const style = isShowLicenseBanner ? { height: BANNER_HEIGHT } : { height: 0 }

  const tips = useMemo(() => {
    if (isExpired) {
      return 'Your subscription is expired.'
    }

    if (isSeatsExceeded) {
      return 'You have more active users than seats included in your subscription.'
    }

    return 'No valid license configured'
  }, [isExpired, isSeatsExceeded])

  return (
    <div
      className={cn(
        'flex items-center justify-between bg-secondary px-4 text-secondary-foreground transition-[height,opacity] md:px-5',
        {
          'opacity-100 pointer-events-auto border-b': isShowLicenseBanner,
          'opacity-0 pointer-events-none': !isShowLicenseBanner
        }
      )}
      style={style}
    >
      <div className="flex items-center gap-1 font-semibold text-destructive">
        <IconNotice />
        {tips}
      </div>

      <div className="flex items-center gap-x-4 md:gap-x-8">
        {pathname !== '/settings/subscription' && (
          <Link
            href="/settings/subscription"
            className={cn(buttonVariants(), 'gap-1')}
          >
            See more
          </Link>
        )}

        <IconClose
          className="cursor-pointer transition-all hover:opacity-70"
          onClick={() => setIsShowLicenseBanner(false)}
        />
      </div>
    </div>
  )
}
