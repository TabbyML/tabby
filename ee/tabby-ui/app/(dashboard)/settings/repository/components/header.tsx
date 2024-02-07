import { SubHeader } from '@/components/sub-header'

export const RepositoryHeader = ({ className }: { className?: string }) => {
  return (
    <SubHeader
      className={className}
      externalLink="https://tabby.tabbyml.com/blog/2023/10/16/repository-context-for-code-completion"
    >
      Tabby supports connecting to Git repositories and uses these repositories
      as a context to enhance performance of large language model.
    </SubHeader>
  )
}
