import { Metadata } from 'next'

import { SubHeader } from '@/components/sub-header'

import RepositoryTabsHeader from './components/tabs-header'

const Header = ({ className }: { className?: string }) => {
  return (
    <SubHeader
      className={className}
      // externalLink="https://tabby.tabbyml.com/blog/2023/10/16/repository-context-for-code-completion"
    >
      <span className="text-foreground text-base">
        Connect to a range of context sources to enhance performance of large
        language model.
      </span>
    </SubHeader>
  )
}

export default function GitLayout({ children }: { children: React.ReactNode }) {
  return (
    <>
      <Header />
      <RepositoryTabsHeader />
      <div className="mt-4">{children}</div>
    </>
  )
}

export const metadata: Metadata = {
  title: 'Repository Providers'
}
