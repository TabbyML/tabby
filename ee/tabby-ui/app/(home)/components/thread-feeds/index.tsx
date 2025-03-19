'use-client'

import { useEffect } from 'react'
import { motion } from 'framer-motion'
import { useQuery } from 'urql'

import { useAllMembers } from '@/lib/hooks/use-all-members'
import {
  setThreadsTab,
  useAnswerEngineStore
} from '@/lib/stores/answer-engine-store'
import { contextInfoQuery, listMyThreads, listThreads } from '@/lib/tabby/query'
import { cn } from '@/lib/utils'
import { IconSpinner } from '@/components/ui/icons'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import LoadingWrapper from '@/components/loading-wrapper'

import { AllThreadFeeds } from './all-threads'
import { MyThreadFeeds } from './my-threads'
import { ThreadFeedsContext } from './threads-context'

interface ThreadFeedsProps {
  className?: string
  onNavigateToThread: () => void
}

export function ThreadFeeds({
  className,
  onNavigateToThread
}: ThreadFeedsProps) {
  const threadsTab = useAnswerEngineStore(state => state.threadsTab)
  const [allUsers, fetchingUsers] = useAllMembers()
  const [{ data: contextInfoData, fetching: fetchingSources }] = useQuery({
    query: contextInfoQuery
  })

  const [{ data: persistedThreads, fetching: fetchingPersistedThreads }] =
    useQuery({
      query: listThreads,
      variables: {
        last: 25,
        isEphemeral: false,
        before: undefined
      }
    })
  const [{ data: myThreads, fetching: fetchingMyThreads }] = useQuery({
    query: listMyThreads,
    variables: {
      last: 25,
      before: undefined
    }
  })

  const loading =
    fetchingPersistedThreads ||
    fetchingMyThreads ||
    fetchingSources ||
    fetchingUsers
  const hasPersistedThreads = !!persistedThreads?.threads?.edges?.length
  const hasMyThreads = !!myThreads?.myThreads.edges.length

  useEffect(() => {
    if (loading) return
    if (!hasPersistedThreads && !hasMyThreads) return

    if (!hasPersistedThreads && threadsTab === 'all') {
      setThreadsTab('mine')
    } else if (!hasMyThreads && threadsTab === 'mine') {
      setThreadsTab('all')
    }
  }, [loading, hasPersistedThreads, hasMyThreads, threadsTab])

  // if there's no thread, hide the section
  if (!hasPersistedThreads && !hasMyThreads) return null

  return (
    <ThreadFeedsContext.Provider
      value={{
        allUsers,
        fetchingUsers,
        sources: contextInfoData?.contextInfo.sources,
        fetchingSources,
        onNavigateToThread
      }}
    >
      <div className={cn('w-full', className)}>
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{
            once: true
          }}
          transition={{
            ease: 'easeOut',
            delay: 0.3
          }}
        >
          <LoadingWrapper
            loading={loading}
            fallback={
              <div className="flex justify-center">
                <IconSpinner className="h-8 w-8" />
              </div>
            }
            delay={600}
          >
            <>
              <Tabs
                value={threadsTab}
                onValueChange={v => setThreadsTab(v as 'all' | 'mine')}
              >
                <div className="flex items-center justify-between pb-3">
                  <TabsList className="w-full justify-start border-b bg-transparent p-0">
                    {!!hasPersistedThreads && (
                      <TabsTrigger
                        value="all"
                        className="rounded-none border-b-2 border-b-transparent bg-transparent px-3 py-2 text-base font-medium shadow-none data-[state=active]:border-b-primary data-[state=active]:text-foreground data-[state=active]:shadow-none"
                      >
                        Recent Activities
                      </TabsTrigger>
                    )}
                    {!!hasMyThreads && (
                      <TabsTrigger
                        value="mine"
                        className="rounded-none border-b-2 border-b-transparent bg-transparent px-3 py-2 text-base font-medium shadow-none data-[state=active]:border-b-primary data-[state=active]:text-foreground data-[state=active]:shadow-none"
                      >
                        My Activities
                      </TabsTrigger>
                    )}
                  </TabsList>
                </div>
                <TabsContent value="all">
                  <AllThreadFeeds />
                </TabsContent>
                <TabsContent value="mine">
                  <MyThreadFeeds />
                </TabsContent>
              </Tabs>
            </>
          </LoadingWrapper>
        </motion.div>
      </div>
    </ThreadFeedsContext.Provider>
  )
}
