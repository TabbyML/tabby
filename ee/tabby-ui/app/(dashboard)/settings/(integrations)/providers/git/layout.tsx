import { SubHeader } from '@/components/sub-header'

export default function GitProviderLayout({
  children
}: {
  children: React.ReactNode
}) {
  return (
    <>
      <SubHeader externalLink="https://tabby.tabbyml.com/blog/2023/10/16/repository-context-for-code-completion">
        Connect to remote and local Git repositories, utilizing these
        repositories as context to enhance the performance of large language
        models.
      </SubHeader>
      {children}
    </>
  )
}
