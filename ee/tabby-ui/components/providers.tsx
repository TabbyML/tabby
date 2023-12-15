'use client'

import * as React from 'react'
import { ThemeProviderProps } from 'next-themes/dist/types'

import { AuthProvider } from '@/lib/tabby/auth'
import { TooltipProvider } from '@/components/ui/tooltip'

export function Providers({ children, ...props }: ThemeProviderProps) {
  return (
    <TooltipProvider>
      <AuthProvider>{children}</AuthProvider>
    </TooltipProvider>
  )
}
