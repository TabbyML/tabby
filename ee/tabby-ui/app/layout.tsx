import { Metadata } from 'next'

import { Toaster } from '@/components/ui/sonner'

import '@/app/globals.css'

import { fontMono, fontSans } from '@/lib/fonts'
import { cn } from '@/lib/utils'
import { DemoBanner } from '@/components/demo-banner'
import { Providers } from '@/components/providers'
import { TailwindIndicator } from '@/components/tailwind-indicator'

export const metadata: Metadata = {
  title: {
    default: 'Tabby',
    template: `Tabby - %s`
  },
  description: 'Tabby, an opensource, self-hosted AI coding assistant.',
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: 'white' },
    { media: '(prefers-color-scheme: dark)', color: 'black' }
  ]
}

interface RootLayoutProps {
  children: React.ReactNode
}

export default function RootLayout({ children }: RootLayoutProps) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head />
      <body
        className={cn(
          'font-sans antialiased',
          fontSans.variable,
          fontMono.variable
        )}
      >
        <Providers attribute="class" defaultTheme="system" enableSystem>
          <div className="flex min-h-screen flex-col">
            <DemoBanner />
            {children}
          </div>
          <Toaster richColors closeButton />
          <TailwindIndicator />
        </Providers>
      </body>
    </html>
  )
}
