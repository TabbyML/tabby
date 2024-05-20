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

export function Providers({ children, ...props }: ThemeProviderProps) {
  return (
    <NextThemesProvider {...props}>
      <UrqlProvider value={client}>
        <TooltipProvider>
          <AuthProvider>
            <TopbarProgressProvider>
              <ShowDemoBannerProvider>
                <EnsureSignin />
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
  const pathname = usePathname()
  const excludePath = ['/chat']

  if (excludePath.includes(pathname)) {
    return <></>
  }

  useAuthenticatedSession()
  return <></>
}
