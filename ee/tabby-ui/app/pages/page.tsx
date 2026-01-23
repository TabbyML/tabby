import { Metadata } from 'next'
import { redirect } from 'next/navigation'

import { ENABLE_CHAT } from '@/lib/constants'

import { Page as PageComponent } from './components/page-main'

export const metadata: Metadata = {
  title: 'Pages'
}

export default function Pages() {
  if (ENABLE_CHAT) {
    return <PageComponent />
  }

  redirect('/')
}
