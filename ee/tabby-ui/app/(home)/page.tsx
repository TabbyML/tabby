'use client'

import {
  UIEventHandler,
  useEffect,
  useLayoutEffect,
  useRef,
  useState
} from 'react'
import Image from 'next/image'
import { useRouter } from 'next/navigation'
import tabbyUrl from '@/assets/logo-dark.png'
import { motion, useAnimationControls } from 'framer-motion'
import { useInView } from 'react-intersection-observer'
import { useQuery } from 'urql'

import { SESSION_STORAGE_KEY } from '@/lib/constants'
import { useHealth } from '@/lib/hooks/use-health'
import { useMe } from '@/lib/hooks/use-me'
import { useIsChatEnabled } from '@/lib/hooks/use-server-info'
import { useStore } from '@/lib/hooks/use-store'
import { useThrottleCallback } from '@/lib/hooks/use-throttle'
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

import { cardVariants } from './components/constants'
import Stats from './components/stats'
import { ThreadFeeds } from './components/thread-feeds'

const SCROLL_THRESHOLD = 5

function MainPanel() {
  const { data: healthInfo } = useHealth()
  const [{ data }] = useMe()
  const isChatEnabled = useIsChatEnabled()
  const disableScroll = useRef(false)
  const [isShowDemoBanner] = useShowDemoBanner()
  const router = useRouter()
  const [isLoading, setIsLoading] = useState(false)
  const resettingScrollPosition = useRef(false)
  const scroller = useRef<HTMLDivElement>(null)
  const lastScrollTop = useRef(0)
  const activitiesRef = useRef<HTMLDivElement>(null)

  const animationControl = useAnimationControls()

  const [{ data: contextInfoData, fetching: fetchingContextInfo }] = useQuery({
    query: contextInfoQuery
  })
  const scrollY = useStore(useScrollStore, state => state.homePage)

  // Prefetch the search page
  useEffect(() => {
    router.prefetch('/search')
  }, [router])

  useLayoutEffect(() => {
    const resetScrollPosition = () => {
      if (scrollY) {
        setTimeout(() => {
          scroller.current?.scrollTo({
            top: Number(scrollY)
          })
          clearHomeScrollPosition()
        })
      }
    }

    if (resettingScrollPosition.current) return
    resetScrollPosition()
    resettingScrollPosition.current = true
  }, [])

  const onSearch = (question: string, ctx?: ThreadRunContexts) => {
    setIsLoading(true)
    sessionStorage.setItem(SESSION_STORAGE_KEY.SEARCH_INITIAL_MSG, question)
    sessionStorage.setItem(
      SESSION_STORAGE_KEY.SEARCH_INITIAL_CONTEXTS,
      JSON.stringify(ctx)
    )
    router.push('/search')
  }

  const throttledScrollHandler = useThrottleCallback(() => {
    if (!scroller.current || !activitiesRef.current || disableScroll.current) {
      return
    }

    const { scrollTop: currentScrollTop, clientHeight } = scroller.current
    const offsetTop = activitiesRef.current.offsetTop
    const isScrollingDown = currentScrollTop > lastScrollTop.current
    const scrollDifference = currentScrollTop - lastScrollTop.current

    lastScrollTop.current = currentScrollTop
    if (Math.abs(scrollDifference) <= SCROLL_THRESHOLD) return

    if (
      isScrollingDown &&
      currentScrollTop < offsetTop &&
      currentScrollTop + clientHeight - offsetTop > 100
    ) {
      disableScroll.current = true

      // scroll to threads
      disableScroll.current = true
      animationControl.start('hidden')
      scroller.current?.scrollTo({
        top: offsetTop,
        behavior: 'smooth'
      })

      setTimeout(() => {
        disableScroll.current = false
      }, 1000)
    }
  }, 50)

  const [ref, isInView] = useInView({
    threshold: 0.1
  })

  const handleScroll: UIEventHandler<HTMLDivElement> = e => {
    if (disableScroll.current) {
      e.preventDefault()
      return
    }
    throttledScrollHandler.run()
  }

  useEffect(() => {
    if (isInView) {
      animationControl.stop()
      animationControl.start('onscreen')
    }
  }, [isInView])

  useEffect(() => {
    disableScroll.current = false

    return () => {
      disableScroll.current = false
    }
  }, [])

  const style = isShowDemoBanner
    ? { height: `calc(100vh - ${BANNER_HEIGHT})` }
    : { height: '100vh' }

  if (!healthInfo || !data?.me) return <></>

  return (
    <ScrollArea style={style} ref={scroller} onScroll={handleScroll}>
      <header className="fixed right-0 top-0 z-10 flex h-16 items-center justify-end px-4 lg:px-10">
        <div className="flex items-center gap-x-6">
          <ClientOnly>
            <ThemeToggle />
          </ClientOnly>
          <UserPanel showHome={false} showSetting>
            <MyAvatar className="h-10 w-10 border" />
          </UserPanel>
        </div>
      </header>

      <main className="flex-col items-center justify-center overflow-auto pb-8 pt-16 lg:flex lg:pb-0">
        <div className="mx-auto flex min-h-0 w-full flex-col items-center px-10 lg:max-w-4xl lg:px-0">
          <motion.div
            className="flex w-full flex-col items-center gap-6"
            initial="initial"
            // whileInView='onscreen'
            animate={animationControl}
            transition={{
              staggerChildren: 0.05
            }}
            viewport={{
              amount: 'some'
            }}
            ref={ref}
          >
            <motion.div variants={cardVariants}>
              <Image
                src={tabbyUrl}
                alt="logo"
                width={192}
                className={cn('mt-4 invert dark:invert-0', {
                  'mb-4': isChatEnabled,
                  'mb-2': !isChatEnabled
                })}
              />
            </motion.div>
            <motion.div variants={cardVariants} className="w-full">
              {isChatEnabled && (
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
              )}
            </motion.div>
            <Stats />
          </motion.div>

          <ThreadFeeds
            className="pt-6"
            ref={activitiesRef}
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
