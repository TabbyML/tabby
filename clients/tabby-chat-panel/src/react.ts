import type { RefObject } from 'react'
import { useEffect, useState } from 'react'

import type { ClientApi, ClientApiMethods, ServerApi } from './index'
import { createClient, createServer } from './index'

function useClient(iframeRef: RefObject<HTMLIFrameElement>, api: ClientApiMethods) {
  const [client, setClient] = useState<ServerApi | null>(null)
  let isCreated = false

  useEffect(() => {
    if (iframeRef.current && !isCreated) {
      isCreated = true
      createClient(iframeRef.current!, api).then(setClient)
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
      createServer(api).then(setServer)
    }
  }, [])

  return server
}

export {
  useClient,
  useServer,
}
