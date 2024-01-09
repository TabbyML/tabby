import { Header } from '@/components/header'

import { SourceCodeBrowser } from './components/source-code-browser'

export default function Page() {
  return (
    <div className="flex flex-col h-screen">
      <Header />
      <SourceCodeBrowser className="flex-1 overflow-hidden" />
    </div>
  )
}
