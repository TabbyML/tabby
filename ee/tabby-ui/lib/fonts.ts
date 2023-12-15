import { JetBrains_Mono as FontMono, Inter as FontSans, Azeret_Mono as FontLogo } from 'next/font/google'

export const fontSans = FontSans({
  subsets: ['latin'],
  variable: '--font-sans'
})

export const fontMono = FontMono({
  subsets: ['latin'],
  variable: '--font-mono'
})

export const fontLogo = FontLogo({
  subsets: ['latin'],
  variable: '--font-logo'
})