import type { RefObject } from 'react'
import { useEffect, useRef } from 'react'

import type { ClientApi, ServerApi } from './index'
import { createClient, createServer } from './index'

function useClient(iframeRef: RefObject<HTMLIFrameElement>, api: ClientApi) {
  const clientRef = useRef<ServerApi | null>(null)

  useEffect(() => {
    if (iframeRef.current && !clientRef.current)
      clientRef.current = createClient(iframeRef.current, api)
  }, [iframeRef.current])

  return clientRef.current
}

function useServer(api: ServerApi) {
  const serverRef = useRef<ClientApi | null>(null)

  useEffect(() => {
    const isInIframe = window.self !== window.top
    if (isInIframe && !serverRef.current)
      serverRef.current = createServer(api)
  }, [])

  return serverRef.current
}

export {
  useClient,
  useServer,
}
