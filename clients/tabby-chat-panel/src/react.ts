import type { RefObject } from 'react'
import { useEffect, useMemo, useState } from 'react'

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
  let isCreated = false

  useEffect(() => {
    setIsInIframe(window.self !== window.top)
  }, [])

  return useMemo(() => {
    if (isInIframe && !isCreated) {
      isCreated = true
      return createServer(api)
    }
  }, [isInIframe])
}

export {
  useClient,
  useServer,
}
