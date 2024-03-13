'use client'

import * as React from 'react'
import { ThemeProvider as NextThemesProvider } from 'next-themes'
import { ThemeProviderProps } from 'next-themes/dist/types'
import { Provider as UrqlProvider } from 'urql'

import { AuthProvider, useAuthenticatedSession } from '@/lib/tabby/auth'
import { client } from '@/lib/tabby/gql'
import { TooltipProvider } from '@/components/ui/tooltip'

import { TopbarProgressProvider } from './topbar-progress-indicator'

export function Providers({ children, ...props }: ThemeProviderProps) {
  return (
    <NextThemesProvider {...props}>
      <UrqlProvider value={client}>
        <TooltipProvider>
          <AuthProvider>
            <TopbarProgressProvider>
              <EnsureSignin />
              {children}
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
