'use client'

import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import Image from 'next/image'
import { useRouter } from 'next/navigation'
import tabbyUrl from '@/assets/logo-dark.png'
import { useQuery } from 'urql'
import { useStore } from 'zustand'

import { useMe } from '@/lib/hooks/use-me'
import { useSelectedModel } from '@/lib/hooks/use-models'
import { useSelectedRepository } from '@/lib/hooks/use-repositories'
import {
  useIsChatEnabled,
  useIsFetchingServerInfo
} from '@/lib/hooks/use-server-info'
import {
  updatePendingUserMessage,
  updateSelectedModel,
  updateSelectedRepoSourceId
} from '@/lib/stores/chat-actions'
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
import { NotificationBox } from '@/components/notification-box'
import SlackDialog from '@/components/slack-dialog'
import TextAreaSearch from '@/components/textarea-search'
import { ThemeToggle } from '@/components/theme-toggle'
import { MyAvatar } from '@/components/user-avatar'
import UserPanel from '@/components/user-panel'

import { AnimationWrapper } from './components/animation-wrapper'
import { RelatedQuestions } from './components/related-questions'
import Stats from './components/stats'
import { ThreadFeeds } from './components/thread-feeds'

// const ThreadFeeds = lazy(() => import('./components/thread-feeds').then(module => ({ default: module.ThreadFeeds })))

function MainPanel() {
  const resettingScroller = useRef(false)
  const scroller = useRef<HTMLDivElement>(null)
  const [{ data }] = useMe()
  const isFetchingServerInfo = useIsFetchingServerInfo()
  const isChatEnabled = useIsChatEnabled()
  const [isShowDemoBanner] = useShowDemoBanner()
  const elementRef = useRef<HTMLDivElement | null>(null)
  const router = useRouter()
  const [isLoading, setIsLoading] = useState(false)
  const [{ data: contextInfoData, fetching: fetchingContextInfo }] = useQuery({
    query: contextInfoQuery
  })
  const scrollY = useStore(useScrollStore, state => state.homePage)

  const { selectedModel, isFetchingModels, models } = useSelectedModel()
  const { selectedRepository, isFetchingRepositories } = useSelectedRepository()

  const showMainSection = !!data?.me || !isFetchingServerInfo

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

  const handleSelectModel = (model: string) => {
    updateSelectedModel(model)
  }

  const onSelectedRepo = (sourceId: string | undefined) => {
    updateSelectedRepoSourceId(sourceId)
  }

  const onSearch = (question: string, context?: ThreadRunContexts) => {
    setIsLoading(true)
    updatePendingUserMessage({
      content: question,
      context
    })
    router.push('/search')
  }

  const onClickRelatedQuestion = (question: string, sourceId: string) => {
    updateSelectedRepoSourceId(sourceId)
    updatePendingUserMessage({
      content: question,
      context: {
        docSourceIds: [sourceId as string],
        codeSourceIds: [sourceId as string],
        modelName: selectedModel
      }
    })
    router.push('/search')
  }

  const style = isShowDemoBanner
    ? { height: `calc(100vh - ${BANNER_HEIGHT})` }
    : { height: '100vh' }

  return (
    <ScrollArea style={style} ref={scroller}>
      <header
        className="transition-top fixed right-0 z-10 flex h-16 items-center justify-end px-4 lg:px-10"
        style={{
          top: isShowDemoBanner ? BANNER_HEIGHT : 0
        }}
      >
        <div className="flex items-center gap-x-6">
          <ClientOnly>
            <ThemeToggle />
          </ClientOnly>
          <NotificationBox />
          <UserPanel showHome={false} showSetting>
            <MyAvatar className="h-10 w-10 border" />
          </UserPanel>
        </div>
      </header>

      {showMainSection && (
        <main
          className="flex-col items-center justify-center pb-4 pt-16 lg:flex"
          ref={elementRef}
        >
          <div className="mx-auto flex w-full flex-col items-center gap-6 px-10 lg:-mt-[2vh] lg:max-w-4xl lg:px-0">
            <AnimationWrapper
              viewport={{
                amount: 0.1
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
                viewport={{
                  amount: 0.1
                }}
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
                  modelName={selectedModel}
                  onSelectModel={handleSelectModel}
                  repoSourceId={selectedRepository?.sourceId}
                  onSelectRepo={onSelectedRepo}
                  isInitializingResources={
                    isFetchingModels || isFetchingRepositories
                  }
                  models={models}
                />
                <RelatedQuestions
                  sourceId={selectedRepository?.sourceId}
                  onClickQuestion={onClickRelatedQuestion}
                />
              </AnimationWrapper>
            )}
            <Stats />
            <ThreadFeeds
              className="lg:mt-8"
              onNavigateToThread={() => {
                if (!scroller.current) return
                setHomeScrollPosition(scroller.current.scrollTop)
              }}
            />
          </div>
        </main>
      )}
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
