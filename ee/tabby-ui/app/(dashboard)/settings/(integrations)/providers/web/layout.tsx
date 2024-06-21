import { SubHeader } from '@/components/sub-header'

export default function WebProviderLayout({
  children
}: {
  children: React.ReactNode
}) {
  return (
    <>
      <SubHeader
      // FIXME(wsxiaoys): add external link
      // externalLink="https://tabby.tabbyml.com/blog/2023/10/16/repository-context-for-code-completion"
      >
        <p>
          Crawl documents from following URLs and use their content to enhance
          the Answer Engine. Recrawling will occur only if manually initiated.
        </p>
        <p>
          Underlying,{' '}
          <a
            className="underline"
            target="_blank"
            href="https://github.com/projectdiscovery/katana"
          >
            Katana
          </a>{' '}
          is used as a crawler (running as a subprocess) and thus needs to be
          installed in the $PATH.
        </p>
      </SubHeader>
      {children}
    </>
  )
}
