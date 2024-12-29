'use client'

import { useEffect, useRef } from 'react'
import { usePathname } from 'next/navigation'

import { ScrollArea } from '@/components/ui/scroll-area'
import { SidebarInset } from '@/components/ui/sidebar'
import { BANNER_HEIGHT, useShowDemoBanner } from '@/components/demo-banner'
import { Header } from '@/components/header'
import { useShowLicenseBanner } from '@/components/license-banner'

export default function MainContent({
  children
}: {
  children: React.ReactNode
}) {
  const pathname = usePathname()
  const scroller = useRef<HTMLDivElement>(null)
  const [isShowDemoBanner] = useShowDemoBanner()
  const [isShowLicenseBanner] = useShowLicenseBanner()
  const style =
    isShowDemoBanner || isShowLicenseBanner
      ? {
          height: `calc(100vh - ${
            isShowDemoBanner ? BANNER_HEIGHT : '0rem'
          } - ${isShowLicenseBanner ? BANNER_HEIGHT : '0rem'})`
        }
      : { height: '100vh' }

  useEffect(() => {
    if (pathname && scroller.current) {
      scroller.current.scrollTop = 0
    }
  }, [pathname])

  return (
    <SidebarInset className="overflow-x-hidden">
      {/* Wraps right hand side into ScrollArea, making scroll bar consistent across all browsers */}
      <ScrollArea ref={scroller} style={style}>
        <Header />
        <div className="p-4 lg:p-10">{children}</div>
      </ScrollArea>
    </SidebarInset>
  )
}
