'use client'

import { BANNER_HEIGHT, useShowDemoBanner } from '@/components/demo-banner'
import { Header } from '@/components/header'
import { useShowLicenseBanner } from '@/components/license-banner'
import { SidebarInset } from '@/components/ui/sidebar'

export default function MainContent({
  children
}: {
  children: React.ReactNode
}) {
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

  return (
    <>
      {/* Wraps right hand side into ScrollArea, making scroll bar consistent across all browsers */}
      <SidebarInset
        className={'flex flex-1 flex-col transition-all overflow-y-auto'}
        style={style}
      >
        <Header />
        <div className="flex-1 p-4 lg:p-10">{children}</div>
      </SidebarInset>
    </>
  )
}
