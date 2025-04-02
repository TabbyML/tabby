import { Dispatch, SetStateAction, useContext, useMemo } from 'react'
import { useQuery } from 'urql'

import { listMyThreads } from '@/lib/tabby/query'
import { Button } from '@/components/ui/button'
import { IconArrowRight } from '@/components/ui/icons'

import { ChatContext } from './chat-context'
import { ThreadItem } from './thread-item'

const exampleMessages = [
  {
    heading: 'Convert list of string to numbers',
    message: `How to convert a list of string to numbers in python`
  },
  {
    heading: 'How to parse email address',
    message: 'How to parse email address with regex'
  }
]

export function EmptyScreen({
  setInput,
  welcomeMessage,
  setShowHistory,
  onSelectThread
}: {
  setInput: (v: string) => void
  welcomeMessage?: string
  setShowHistory: Dispatch<SetStateAction<boolean>>
  onSelectThread: (threadId: string) => void
}) {
  const { contextInfo, fetchingContextInfo } = useContext(ChatContext)
  const welcomeMsg = welcomeMessage || 'Welcome'
  const [{ data }] = useQuery({
    query: listMyThreads,
    variables: {
      last: 5
    }
  })

  const threads = useMemo(() => {
    return data?.myThreads?.edges?.slice(-5).reverse()
  }, [data?.myThreads?.edges])

  const onNavigateToThread = (threadId: string) => {
    onSelectThread(threadId)
    setShowHistory(false)
  }

  return (
    <div className="mx-auto max-w-5xl">
      <div>
        <h1 className="mb-2 text-2xl font-semibold">{welcomeMsg}</h1>
        <p className="leading-normal text-muted-foreground">
          You can start a conversation here or try the following examples:
        </p>
        <div className="mt-4 flex flex-col items-start space-y-2">
          {exampleMessages.map((message, index) => (
            <Button
              key={index}
              variant="link"
              className="h-auto p-0 text-base"
              onClick={() => setInput(message.message)}
            >
              <IconArrowRight className="mr-2 text-muted-foreground" />
              <p className="text-left">{message.heading}</p>
            </Button>
          ))}
        </div>
      </div>
      {!!threads?.length && (
        <div className="mt-10">
          <div className="mb-3 flex items-center gap-2">
            <span className="text-lg font-semibold">Recent Activities</span>
          </div>
          <div className="space-y-4">
            {threads?.map(x => {
              return (
                <ThreadItem
                  key={x.node.id}
                  data={x}
                  onNavigate={onNavigateToThread}
                  sources={contextInfo?.sources}
                  fetchingSources={fetchingContextInfo}
                />
              )
            })}
          </div>
          <div className="text-center">
            <Button
              size="sm"
              variant="ghost"
              onClick={e => setShowHistory(true)}
              className="mt-4 text-foreground/70"
            >
              View all history
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}
