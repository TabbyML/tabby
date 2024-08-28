'use client'

import { useEffect, useState } from 'react'
import { usePathname, useSearchParams } from 'next/navigation'
import posthog, { PostHog } from 'posthog-js'
import { PostHogProvider as Provider, usePostHog } from 'posthog-js/react'

import { useIsDemoMode } from '@/lib/hooks/use-server-info'

const POSTHOG_KEY = 'phc_aBzNGHzlOy2C8n1BBDtH7d4qQsIw9d8T0unVlnKfdxB'
const POSTHOG_HOST = 'https://us.i.posthog.com'

function PostHogPageView(): null {
  const pathname = usePathname()
  const searchParams = useSearchParams()
  const posthog = usePostHog()
  useEffect(() => {
    if (pathname && posthog) {
      let url = window.origin + pathname
      if (searchParams.toString()) {
        url = url + `?${searchParams.toString()}`
      }
      posthog.capture('$pageview', {
        $current_url: url
      })
    }
  }, [pathname, searchParams, posthog])

  return null
}

export function PostHogProvider({ children }: { children: React.ReactNode }) {
  const isDemoMode = useIsDemoMode()
  const [postHogInstance, setPostHogInstance] = useState<PostHog | undefined>()

  useEffect(() => {
    if (typeof window !== 'undefined' && isDemoMode && !postHogInstance) {
      const isInIframe = window.self !== window.top
      if (isInIframe) return

      const postHogInstance = posthog.init(POSTHOG_KEY, {
        api_host: POSTHOG_HOST,
        person_profiles: 'identified_only',
        capture_pageview: false
      })
      setPostHogInstance(postHogInstance || undefined)
    }
  }, [isDemoMode])

  return (
    <Provider client={postHogInstance}>
      <>
        <PostHogPageView />
        {children}
      </>
    </Provider>
  )
}
