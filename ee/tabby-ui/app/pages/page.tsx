import { Metadata } from 'next'

import { Page as PageComponent } from './components/page-main'

export const metadata: Metadata = {
  title: 'Pages'
}

export default function Pages() {
  return <PageComponent />
}
