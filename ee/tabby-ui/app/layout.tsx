import { Metadata } from 'next'
import { Toaster } from 'react-hot-toast'

import '@/app/globals.css'

import { fontMono, fontSans } from '@/lib/fonts'
import { cn } from '@/lib/utils'
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
        <Toaster />
        <Providers attribute="class" defaultTheme="system" enableSystem>
          <div className="flex min-h-screen flex-col">{children}</div>
          <TailwindIndicator />
        </Providers>
      </body>
    </html>
  )
}
