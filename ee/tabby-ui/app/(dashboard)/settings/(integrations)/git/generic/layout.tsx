import React from 'react'

import { SubHeader } from '@/components/sub-header'

const Header = ({ className }: { className?: string }) => {
  return (
    <SubHeader
      className={className}
      externalLink="https://tabby.tabbyml.com/blog/2023/10/16/repository-context-for-code-completion"
    >
      Connect to Git repositories and uses these repositories as a context to
      enhance performance of large language model.
    </SubHeader>
  )
}

export default function GenericGitRepositoriesLayout({
  children
}: {
  children: React.ReactNode
}) {
  return (
    <>
      <Header />
      {children}
    </>
  )
}
