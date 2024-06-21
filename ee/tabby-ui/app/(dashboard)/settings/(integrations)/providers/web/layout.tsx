import { SubHeader } from '@/components/sub-header'

export default function WebProviderLayout({
  children
}: {
  children: React.ReactNode
}) {
  return (
    <>
      <SubHeader
      // FIXME(meng): add external link
      // externalLink="https://tabby.tabbyml.com/blog/2023/10/16/repository-context-for-code-completion"
      >
        Crawl web docs starting from following URLs, and use these content to enhance output of LLM Answer Engine.
        Underlying tabby uses <a target="_blank" href="https://github.com/projectdiscovery/katana">Katana</a> to crawl web docs in a subprocess, and no recrawl will be done until user manually triggers it.
      </SubHeader>
      {children}
    </>
  )
}
