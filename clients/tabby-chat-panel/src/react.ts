import type { RefObject } from 'react'
import { useEffect, useMemo, useState, useRef } from 'react'
import { Thread } from '@quilted/threads'

import type { Api } from './index'
import { createClient, createServer } from './index'

function useClient(iframeRef: RefObject<HTMLIFrameElement>) {
  return useMemo(() => {
    if (iframeRef.current) {
      return createClient(iframeRef.current)
    }
  }, [iframeRef.current])
}

function useServer(api: Api) {
  const [isInIframe, setIsInIframe] = useState(false)
  const serverRef = useRef<Thread<Record<string, never>> | null>(null)

  useEffect(() => {
    const isInIframe = window.self !== window.top
    if (isInIframe && !serverRef.current) {
      serverRef.current = createServer(api)
    }
    setIsInIframe(isInIframe)
  }, [])

  return useMemo(() => {
    return serverRef.current
  }, [isInIframe])
}

export {
  useClient,
  useServer,
}
