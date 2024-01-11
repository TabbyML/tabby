import { Header } from '@/components/header'

import { SourceCodeBrowser } from './components/source-code-browser'

export default function Page() {
  return (
    <div className="flex h-screen flex-col">
      <Header />
      <SourceCodeBrowser className="flex-1 overflow-hidden" />
    </div>
  )
}
