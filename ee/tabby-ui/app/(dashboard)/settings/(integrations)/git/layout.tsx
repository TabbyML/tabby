import GitTabsHeader from './components/tabs-header'

export default function GitLayout({ children }: { children: React.ReactNode }) {
  return (
    <>
      <GitTabsHeader />
      {children}
    </>
  )
}
