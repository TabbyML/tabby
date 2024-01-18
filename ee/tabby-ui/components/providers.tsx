'use client'

import * as React from 'react'
import { ThemeProvider as NextThemesProvider } from 'next-themes'
import { ThemeProviderProps } from 'next-themes/dist/types'

import { AuthProvider, useAuthenticatedSession } from '@/lib/tabby/auth'
import { TooltipProvider } from '@/components/ui/tooltip'
import { Provider as UrqlProvider } from 'urql'
import { client } from '@/lib/tabby/urql'

export function Providers({ children, ...props }: ThemeProviderProps) {
  return (
    <NextThemesProvider {...props}>
      <UrqlProvider value={client}>
        <TooltipProvider>
          <AuthProvider>
            <EnsureSignin />
            {children}
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
