'use client'

import { useHydrated } from '@/lib/hooks/use-hydration'
import {
  toggleSidebar,
  useUserPreferencesStore
} from '@/lib/stores/user-preferences-store'
import { SidebarProvider } from '@/components/ui/sidebar'

import MainContent from './dashboard-main'
import AppSidebar from './dashboard-sidebar'

export default function Layout({ children }: { children: React.ReactNode }) {
  const hydrated = useHydrated()
  const isSidebarExpanded = useUserPreferencesStore(
    state => state.isSidebarExpanded
  )

  if (!hydrated) return null

  return (
    <>
      <SidebarProvider
        className="relative"
        open={isSidebarExpanded}
        onOpenChange={open => toggleSidebar(open)}
      >
        <AppSidebar />
        <MainContent>{children}</MainContent>
      </SidebarProvider>
    </>
  )
}
