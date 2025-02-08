import { Metadata } from 'next'

import { Page as PageComponent } from './components/page-main'

export const metadata: Metadata = {
  title: 'Page'
}

export default function Page() {
  return <PageComponent />
}
