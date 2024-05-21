'use client'

import * as React from 'react'
import { usePathname } from 'next/navigation'
import { ThemeProvider as NextThemesProvider } from 'next-themes'
import { ThemeProviderProps } from 'next-themes/dist/types'
import { Provider as UrqlProvider } from 'urql'

import { AuthProvider, useAuthenticatedSession } from '@/lib/tabby/auth'
import { client } from '@/lib/tabby/gql'
import { TooltipProvider } from '@/components/ui/tooltip'
import { ShowDemoBannerProvider } from '@/components/demo-banner'

import { TopbarProgressProvider } from './topbar-progress-indicator'

const publicPaths = ['/chat']

export function Providers({ children, ...props }: ThemeProviderProps) {
  const pathName = usePathname()
  const isPublicPath = publicPaths.includes(pathName)
  return (
    <NextThemesProvider {...props}>
      <UrqlProvider value={client}>
        <TooltipProvider>
          <AuthProvider>
            <TopbarProgressProvider>
              <ShowDemoBannerProvider>
                {!isPublicPath && <EnsureSignin />}
                {children}
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
