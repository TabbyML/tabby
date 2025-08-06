'use client'

import Link from 'next/link'
import logoDarkUrl from '@/assets/logo-dark.png'
import logoUrl from '@/assets/logo.png'

import { cn } from '@/lib/utils'

import { BrandingLogo } from './branding-logo'
import { ClientOnly } from './client-only'
import { BANNER_HEIGHT, useShowDemoBanner } from './demo-banner'
import { ThemeToggle } from './theme-toggle'
import { buttonVariants } from './ui/button'
import { MyAvatar } from './user-avatar'
import UserPanel from './user-panel'

export default function NotFoundPage() {
  const [isShowDemoBanner] = useShowDemoBanner()

  const style = isShowDemoBanner
    ? {
        height: `calc(100vh - ${BANNER_HEIGHT})`
      }
    : { height: '100vh' }

  return (
    <div style={style} className="flex flex-col">
      <Header />
      <div className="flex flex-1 flex-col items-center justify-center">
        <h2 className="mt-4 text-6xl font-bold tracking-tight text-foreground sm:text-7xl">
          404
        </h2>
        <p className="mt-4 text-lg text-muted-foreground">
          Oops, it looks like the page you&apos;re looking for doesn&apos;t
          exist.
        </p>
        <Link className={cn('mt-6', buttonVariants())} href="/">
          Home
        </Link>
      </div>
    </div>
  )
}

function Header() {
  return (
    <header className="flex h-16 items-center justify-between border-b px-4 lg:px-10">
      <div className="flex items-center">
        <Link href="/">
          <BrandingLogo
            defaultLogoUrl={logoUrl.src}
            alt="logo"
            width={128}
            className="dark:hidden"
          />
          <BrandingLogo
            defaultLogoUrl={logoDarkUrl.src}
            alt="logo"
            width={96}
            className="hidden dark:block"
          />
        </Link>
      </div>
      <div className="flex items-center gap-6">
        <ClientOnly>
          <ThemeToggle />
        </ClientOnly>
        <UserPanel showSetting>
          <MyAvatar className="h-10 w-10 border" />
        </UserPanel>
      </div>
    </header>
  )
}
