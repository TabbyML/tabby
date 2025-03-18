import { type RefObject, useEffect, useState } from 'react'
import { createThreadFromIframe, createThreadFromInsideIframe } from 'tabby-threads'
import { createClient, createServer } from './thread'
import type { ClientApi } from './client'
import type { ServerApi, ServerApiList } from './server'

export function useClient(iframeRef: RefObject<HTMLIFrameElement>, api: ClientApi) {
  const [client, setClient] = useState<ServerApiList | null>(null)
  let isCreated = false

  useEffect(() => {
    if (iframeRef.current && !isCreated) {
      isCreated = true
      const thread = createThreadFromIframe<ClientApi, ServerApi>(iframeRef.current, { expose: api })
      createClient(thread).then(setClient)
    }
  }, [iframeRef.current])

  return client
}

export function useServer(api: ServerApi) {
  const [server, setServer] = useState<ClientApi | null>(null)
  let isCreated = false

  useEffect(() => {
    const isInIframe = window.self !== window.top
    if (isInIframe && !isCreated) {
      isCreated = true
      const thread = createThreadFromInsideIframe<ServerApi, ClientApi>({ expose: api })
      createServer(thread).then(setServer)
    }
  }, [])

  return server
}
