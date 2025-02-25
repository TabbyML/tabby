'use client'

import * as React from 'react'
import { useInView } from 'react-intersection-observer'

import { useAtBottom } from '@/lib/hooks/use-at-bottom'

import { ChatContext } from './chat-context'

interface ChatScrollAnchorProps {
  trackVisibility?: boolean
}

export function ChatScrollAnchor({ trackVisibility }: ChatScrollAnchorProps) {
  const { container } = React.useContext(ChatContext)
  const isAtBottom = useAtBottom(100, container)

  const { ref, entry, inView } = useInView({
    trackVisibility,
    delay: 100,
    rootMargin: '0px 0px -150px 0px',
    root: container
  })

  React.useEffect(() => {
    if (isAtBottom && trackVisibility && !inView) {
      entry?.target.scrollIntoView({
        block: 'start'
      })
    }
  }, [inView, entry, isAtBottom, trackVisibility])

  return <div ref={ref} className="h-px w-full" />
}
