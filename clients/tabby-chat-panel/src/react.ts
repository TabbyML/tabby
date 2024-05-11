import type { RefObject } from 'react'
import { useEffect, useMemo, useState } from 'react'
import { createThreadFromIframe, createThreadFromInsideIframe } from '@quilted/threads'

import type { Api } from './index'
import { createClient, createServer } from './index'

function useClient(iframeRef: RefObject<HTMLIFrameElement>) {
  return useMemo(() => {
    if (iframeRef.current)
      return createClient(createThreadFromIframe, iframeRef.current)
  }, [iframeRef.current])
}

function useServer(api: Api) {
  const [isInIframe, setIsInIframe] = useState(false)

  useEffect(() => {
    setIsInIframe(window.self !== window.top)
  }, [])

  return useMemo(() => {
    if (isInIframe)
      return createServer(createThreadFromInsideIframe, api)
  }, [isInIframe])
}

export {
  useClient,
  useServer,
}
