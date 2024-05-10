import { fromIframe, fromInsideIframe } from "@remote-ui/rpc"
import { RefObject, useMemo } from "react"
import { Api, createClient, createServer } from "./index"

function useClient(iframeRef: RefObject<HTMLIFrameElement>) {
  return useMemo(() => {
    if (iframeRef.current)
      return createClient(fromIframe(iframeRef.current))
  }, [iframeRef.current])
}

function useServer(api: Api) {
  return useMemo(() => {
    return createServer(fromInsideIframe(), api)
  }, [])
}

export {
  useClient,
  useServer,
}
