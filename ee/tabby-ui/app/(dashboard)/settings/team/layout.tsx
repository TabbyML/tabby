import { TeamNav } from './components/team-nav'

export default function TeamLayout({
  children
}: {
  children: React.ReactNode
}) {
  return (
    <>
      <TeamNav />
      {children}
    </>
  )
}
