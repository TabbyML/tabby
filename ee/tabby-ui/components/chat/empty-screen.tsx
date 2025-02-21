import { Dispatch, SetStateAction } from 'react'
import { useQuery } from 'urql'

import { listThreads } from '@/lib/tabby/query'
import { Button } from '@/components/ui/button'
import { IconArrowRight } from '@/components/ui/icons'

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
  setShowHistory
}: {
  setInput: (v: string) => void
  welcomeMessage?: string
  setShowHistory: Dispatch<SetStateAction<boolean>>
}) {
  const welcomeMsg = welcomeMessage || 'Welcome'
  const [{ data, fetching }] = useQuery({
    // todo  -> myThreads
    query: listThreads,
    variables: {
      last: 5
    }
  })

  const threads = data?.threads?.edges?.slice(0, 5)

  const onNavigateToThread = () => {
    // 
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
      {/* todo conditions */}
      {/* {!!threads?.length && ( */}
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
                sources={undefined}
              />
            )
          })}
        </div>
        <div className="text-center">
          <Button
            size="sm"
            variant="ghost"
            onClick={e => setShowHistory(true)}
            className='text-foreground/70 mt-4'
          >
            View all history
          </Button>
        </div>
      </div>
      {/* )} */}
    </div>
  )
}
