import * as React from 'react'
import { createRoot } from 'react-dom/client'

import { useClient } from './react'

declare global {
  interface Window {
    token?: string
    endpoint?: string
  }
}

function ChatPanel() {
  const [endpoint, setEndpoint] = React.useState('')
  const [token, setToken] = React.useState('')
  const iframeRef = React.useRef<HTMLIFrameElement>(null)

  const client = useClient(iframeRef, {
    navigate: () => {
      // FIXME(wwayne): Send message to VSCode webview to invoke VSCode API
    },
  })

  React.useEffect(() => {
    setEndpoint(window.endpoint || '')
    setToken(window.token || '')
  }, [])

  React.useEffect(() => {
    if (iframeRef?.current && token) {
      client?.init({
        fetcherOptions: {
          authorization: token,
        },
      })
    }
  }, [iframeRef?.current, client, token])

  return (
    <iframe
      src={`${endpoint}/chat`}
      style={{
        width: '100vw',
        height: '100vh',
      }}
      ref={iframeRef}
    />
  )
}

createRoot(document.getElementById('root') as HTMLElement).render(
  <ChatPanel />,
)
