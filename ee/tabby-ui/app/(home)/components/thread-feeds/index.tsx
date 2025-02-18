import { motion } from 'framer-motion'
import { useQuery } from 'urql'

import { useAllMembers } from '@/lib/hooks/use-all-members'
import {
  setThreadsTab,
  useAnswerEngineStore
} from '@/lib/stores/answer-engine-store'
import { contextInfoQuery } from '@/lib/tabby/query'
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
  // FIXME rename
  const threadsTab = useAnswerEngineStore(state => state.threadsTab)
  const [allUsers, fetchingUsers] = useAllMembers()
  const [{ data: contextInfoData, fetching: fetchingSources }] = useQuery({
    query: contextInfoQuery
  })

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
            // FIXME fetching threads
            loading={fetchingSources || fetchingUsers}
            fallback={
              <div className="flex justify-center">
                <IconSpinner className="h-8 w-8" />
              </div>
            }
            delay={600}
          >
            {/* todo hasThreads ? */}
            <>
              {/* <div className="mb-2.5 w-full text-lg font-semibold">
                Recent Activities
              </div>
              <Separator className="mb-4 w-full" /> */}
              <Tabs
                value={threadsTab}
                onValueChange={v => setThreadsTab(v as 'all' | 'mine')}
              >
                <div className="flex items-center justify-between pb-3">
                  <TabsList className="w-full justify-start border-b bg-transparent p-0">
                    <TabsTrigger
                      value="all"
                      className="rounded-none border-b-2 border-b-transparent bg-transparent px-3 py-2 text-base font-medium shadow-none data-[state=active]:border-b-primary data-[state=active]:text-foreground data-[state=active]:shadow-none"
                    >
                      Recent Activities
                    </TabsTrigger>
                    <TabsTrigger
                      value="mine"
                      className="rounded-none border-b-2 border-b-transparent bg-transparent px-3 py-2 text-base font-medium shadow-none data-[state=active]:border-b-primary data-[state=active]:text-foreground data-[state=active]:shadow-none"
                    >
                      My Activities
                    </TabsTrigger>
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
