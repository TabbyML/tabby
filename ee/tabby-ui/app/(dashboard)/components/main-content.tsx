'use client'

import { ScrollArea } from '@/components/ui/scroll-area'
import { BANNER_HEIGHT, useShowDemoBanner } from '@/components/demo-banner'
import { Header } from '@/components/header'

export default function MainContent({
  children
}: {
  children: React.ReactNode
}) {
  const [isShowDemoBanner] = useShowDemoBanner()

  const style = isShowDemoBanner
    ? { height: `calc(100vh - ${BANNER_HEIGHT})` }
    : { height: '100vh' }
  return (
    <>
      {/* Wraps right hand side into ScrollArea, making scroll bar consistent across all browsers */}
      <ScrollArea
        className={'flex flex-1 flex-col transition-all'}
        style={style}
      >
        <Header />
        <div className="flex-1 p-4 lg:p-10">{children}</div>
      </ScrollArea>
    </>
  )
}
