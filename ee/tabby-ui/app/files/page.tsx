import { Metadata } from 'next'

import { SourceCodeBrowser } from './components/source-code-browser'

export const metadata: Metadata = {
  title: 'Code Browser'
}

export default function Page() {
  return (
    <div className="flex h-screen flex-col">
      <SourceCodeBrowser className="flex-1 overflow-hidden" />
    </div>
  )
}
