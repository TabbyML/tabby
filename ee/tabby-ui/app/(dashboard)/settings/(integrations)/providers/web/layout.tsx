import { SubHeader } from '@/components/sub-header'

export default function WebProviderLayout({
  children
}: {
  children: React.ReactNode
}) {
  return (
    <>
      <SubHeader
      // todo: add external link
      // externalLink="https://tabby.tabbyml.com/blog/2023/10/16/repository-context-for-code-completion"
      >
        Connect to a web URL, utilizing this as context to enhance the
        performance of large language models.
      </SubHeader>
      {children}
    </>
  )
}
