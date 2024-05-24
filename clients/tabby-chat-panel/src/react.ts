import type { RefObject } from 'react'
import { useEffect, useState } from 'react'

import type { ClientApi, ServerApi } from './index'
import { createClient, createServer } from './index'

function useClient(iframeRef: RefObject<HTMLIFrameElement>, api: ClientApi) {
  const [client, setClient] = useState<ServerApi | null>(null)
  let isCreated = false

  useEffect(() => {
    if (iframeRef.current && !isCreated) {
      isCreated = true
      setClient(createClient(iframeRef.current, api))
    }
  }, [iframeRef.current])

  return client
}

function useServer(api: ServerApi) {
  const [server, setServer] = useState<ClientApi | null>(null)
  let isCreated = false

  useEffect(() => {
    const isInIframe = window.self !== window.top
    if (isInIframe && !isCreated) {
      isCreated = true
      setServer(createServer(api))
    }
  }, [])

  return server
}

export {
  useClient,
  useServer,
}
