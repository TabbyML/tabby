import {
  JetBrains_Mono as FontMono,
  Montserrat as FontMontserrat,
  Inter as FontSans
} from 'next/font/google'

export const fontSans = FontSans({
  subsets: ['latin'],
  variable: '--font-sans'
})

export const fontMono = FontMono({
  subsets: ['latin'],
  variable: '--font-mono'
})

export const fontMontserrat = FontMontserrat({
  subsets: ['latin'],
  variable: '--font-montserrat',
  weight: '600'
})
