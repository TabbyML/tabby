import { JetBrains_Mono as FontMono, Inter as FontSans } from 'next/font/google'

import localFont from "next/font/local"

export const fontSans = localFont({
  src: "../assets/fonts/Inter-Regular.woff",
  variable: '--font-sans'
})

export const fontMono = FontMono({
  subsets: ['latin'],
  variable: '--font-mono'
})
