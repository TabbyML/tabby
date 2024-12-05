'use client'

import * as React from 'react'
import { usePathname, useSearchParams } from 'next/navigation'
import { ThemeProvider as NextThemesProvider } from 'next-themes'
import { ThemeProviderProps } from 'next-themes/dist/types'
import { Provider as UrqlProvider } from 'urql'

import { PostHogProvider } from '@/lib/posthog'
import { AuthProvider, useAuthenticatedSession } from '@/lib/tabby/auth'
import { client } from '@/lib/tabby/gql'
import { TooltipProvider } from '@/components/ui/tooltip'
import { ShowDemoBannerProvider } from '@/components/demo-banner'

import { ShowLicenseBannerProvider } from './license-banner'
import { TopbarProgressProvider } from './topbar-progress-indicator'

const publicPaths = ['/chat']

export function Providers({ children, ...props }: ThemeProviderProps) {
  const pathName = usePathname()
  const isPublicPath = publicPaths.includes(pathName)
  const searchParams = useSearchParams()
  const themeFromQuery = searchParams.get('theme')
  const clientFromQuery = searchParams.get('client')
  if (themeFromQuery) props.defaultTheme = themeFromQuery
  if (clientFromQuery === 'vscode') {
    // The dark theme's default background color does not match VSCode's theme
    // when rendering in VSCode, we use a separate CSS variable instead of relying on the theme
    props.defaultTheme = 'none'
  }
  return (
    <NextThemesProvider {...props}>
      <UrqlProvider value={client}>
        <TooltipProvider>
          <AuthProvider>
            <TopbarProgressProvider>
              <ShowDemoBannerProvider>
                <ShowLicenseBannerProvider>
                  <PostHogProvider>
                    {!isPublicPath && <EnsureSignin />}
                    {children}
                  </PostHogProvider>
                </ShowLicenseBannerProvider>
              </ShowDemoBannerProvider>
            </TopbarProgressProvider>
          </AuthProvider>
        </TooltipProvider>
      </UrqlProvider>
    </NextThemesProvider>
  )
}

function EnsureSignin() {
  useAuthenticatedSession()
  return <></>
}
