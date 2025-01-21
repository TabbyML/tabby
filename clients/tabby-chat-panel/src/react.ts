import type { RefObject } from 'react'
import { useEffect, useState } from 'react'

import type { ClientApi, ClientApiMethods, ServerApi } from './index'
import { createClient, createServer } from './index'

function useClient(iframeRef: RefObject<HTMLIFrameElement>, api: ClientApiMethods) {
  const [client, setClient] = useState<ServerApi | null>(null)
  let isCreated = false

  useEffect(() => {
    const init = async () => {
      if (iframeRef.current && !isCreated) {
        isCreated = true
        setClient(await createClient(iframeRef.current!, api))
      }
    }
    init()
  }, [iframeRef.current])

  return client
}

function useServer(api: ServerApi) {
  const [server, setServer] = useState<ClientApi | null>(null)
  let isCreated = false

  useEffect(() => {
    const init = async () => {
      const isInIframe = window.self !== window.top
      // eslint-disable-next-line no-console
      console.log('[useServer] isInIframe:', isInIframe)
      if (isInIframe && !isCreated) {
        isCreated = true
        try {
          // eslint-disable-next-line no-console
          console.log('[useServer] Creating server...')
          setServer(await createServer(api))
          // eslint-disable-next-line no-console
          console.log('[useServer] Server created successfully')
        }
        catch (error) {
          // eslint-disable-next-line no-console
          console.error('[useServer] Failed to create server:', error)
          isCreated = false
        }
      }
    }
    // eslint-disable-next-line no-console
    console.log('[useServer] Starting initialization...')
    init()
  }, [])

  return server
}

export {
  useClient,
  useServer,
}
