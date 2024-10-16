'use client'

import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import Image from 'next/image'
import { useRouter } from 'next/navigation'
import tabbyUrl from '@/assets/logo-dark.png'
import { useQuery } from 'urql'

import { SESSION_STORAGE_KEY } from '@/lib/constants'
import { useHealth } from '@/lib/hooks/use-health'
import { useMe } from '@/lib/hooks/use-me'
import { useIsChatEnabled } from '@/lib/hooks/use-server-info'
import { useStore } from '@/lib/hooks/use-store'
import {
  clearHomeScrollPosition,
  setHomeScrollPosition,
  useScrollStore
} from '@/lib/stores/scroll-store'
import { contextInfoQuery } from '@/lib/tabby/query'
import { ThreadRunContexts } from '@/lib/types'
import { cn } from '@/lib/utils'
import { ScrollArea } from '@/components/ui/scroll-area'
import { ClientOnly } from '@/components/client-only'
import { BANNER_HEIGHT, useShowDemoBanner } from '@/components/demo-banner'
import SlackDialog from '@/components/slack-dialog'
import TextAreaSearch from '@/components/textarea-search'
import { ThemeToggle } from '@/components/theme-toggle'
import { MyAvatar } from '@/components/user-avatar'
import UserPanel from '@/components/user-panel'

import { AnimationWrapper } from './components/animation-wrapper'
import Stats from './components/stats'
import { ThreadFeeds } from './components/thread-feeds'

function MainPanel() {
  const resettingScroller = useRef(false)
  const scroller = useRef<HTMLDivElement>(null)
  const { data: healthInfo } = useHealth()
  const [{ data }] = useMe()
  const isChatEnabled = useIsChatEnabled()
  const [isShowDemoBanner] = useShowDemoBanner()
  const elementRef = useRef<HTMLDivElement | null>(null)
  const router = useRouter()
  const [isLoading, setIsLoading] = useState(false)
  const [{ data: contextInfoData, fetching: fetchingContextInfo }] = useQuery({
    query: contextInfoQuery
  })
  const scrollY = useStore(useScrollStore, state => state.homePage)

  // Prefetch the search page
  useEffect(() => {
    router.prefetch('/search')
  }, [router])

  useLayoutEffect(() => {
    const resetScroll = () => {
      if (scrollY) {
        setTimeout(() => {
          scroller.current?.scrollTo({
            top: Number(scrollY)
          })
          clearHomeScrollPosition()
        })
      }
    }

    if (resettingScroller.current) return
    resetScroll()
    resettingScroller.current = true
  }, [])

  if (!healthInfo || !data?.me) return <></>

  const onSearch = (question: string, ctx?: ThreadRunContexts) => {
    setIsLoading(true)
    sessionStorage.setItem(SESSION_STORAGE_KEY.SEARCH_INITIAL_MSG, question)
    sessionStorage.setItem(
      SESSION_STORAGE_KEY.SEARCH_INITIAL_CONTEXTS,
      JSON.stringify(ctx)
    )
    router.push('/search')
  }

  const style = isShowDemoBanner
    ? { height: `calc(100vh - ${BANNER_HEIGHT})` }
    : { height: '100vh' }

  return (
    <ScrollArea style={style} ref={scroller}>
      <header className="sticky top-0 z-10 flex h-16 items-center justify-end bg-background px-4 lg:px-10">
        <div className="flex items-center gap-x-6">
          <ClientOnly>
            <ThemeToggle />
          </ClientOnly>
          <UserPanel showHome={false} showSetting>
            <MyAvatar className="h-10 w-10 border" />
          </UserPanel>
        </div>
      </header>

      <main
        className="flex-col items-center justify-center lg:flex"
        ref={elementRef}
      >
        <div className="mx-auto flex w-full flex-col items-center gap-6 px-10 lg:-mt-[2vh] lg:max-w-4xl lg:px-0">
          <AnimationWrapper
            viewport={{
              margin: '-120px 0px 0px 0px'
            }}
          >
            <Image
              src={tabbyUrl}
              alt="logo"
              width={192}
              className={cn('mt-4 invert dark:invert-0', {
                'mb-4': isChatEnabled,
                'mb-2': !isChatEnabled
              })}
            />
          </AnimationWrapper>
          {isChatEnabled && (
            <AnimationWrapper
              viewport={{ margin: '-140px 0px 0px 0px' }}
              style={{ width: '100%' }}
              delay={0.05}
            >
              <TextAreaSearch
                onSearch={onSearch}
                showBetaBadge
                autoFocus
                loadingWithSpinning
                isLoading={isLoading}
                cleanAfterSearch={false}
                contextInfo={contextInfoData?.contextInfo}
                fetchingContextInfo={fetchingContextInfo}
              />
            </AnimationWrapper>
          )}
          <Stats />
          <ThreadFeeds
            onNavigateToThread={() => {
              if (!scroller.current) return
              setHomeScrollPosition(scroller.current.scrollTop)
            }}
          />
        </div>
      </main>
    </ScrollArea>
  )
}

export default function Home() {
  return (
    <div>
      <MainPanel />
      <SlackDialog />
    </div>
  )
}
