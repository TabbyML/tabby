import type { RefObject } from 'react'
import { useEffect, useState } from 'react'

import type { ClientApi, ServerApi } from './index'
import { createClient, createServer } from './index'

function useClient(iframeRef: RefObject<HTMLIFrameElement>, api: ClientApi) {
  const [client, setClient] = useState<ServerApi | null>(null)
  let abortController: AbortController

  useEffect(() => {
    if (iframeRef.current) {
      abortController?.abort()
      setClient(createClient(iframeRef.current, api, abortController?.signal))
      abortController = new AbortController()
    }
  }, [iframeRef.current])

  return client
}

function useServer(api: ServerApi) {
  const [server, setServer] = useState<ClientApi | null>(null)
  let abortController: AbortController

  useEffect(() => {
    const isInIframe = window.self !== window.top
    if (isInIframe) {
      abortController?.abort()
      setServer(createServer(api, abortController?.signal))
      abortController = new AbortController()
    }
  }, [])

  return server
}

export {
  useClient,
  useServer,
}
