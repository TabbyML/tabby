import Link from 'next/link'

import { ERROR_CODE_NOT_FOUND } from '@/lib/constants'
import { clearHomeScrollPosition } from '@/lib/stores/scroll-store'
import { ExtendedCombinedError } from '@/lib/types'
import { cn } from '@/lib/utils'
import { buttonVariants } from '@/components/ui/button'
import { IconFileSearch } from '@/components/ui/icons'
import NotFoundPage from '@/components/not-found-page'

import { Header } from './header'

interface ErrorViewProps {
  error: ExtendedCombinedError
  pageIdFromURL?: string
}

export function ErrorView({ error, pageIdFromURL }: ErrorViewProps) {
  let title = 'Something went wrong'
  let description = 'Failed to fetch, please refresh the page'

  if (error.message === ERROR_CODE_NOT_FOUND) {
    return <NotFoundPage />
  }

  return (
    <div className="flex h-screen flex-col">
      <Header pageIdFromURL={pageIdFromURL} />
      <div className="flex-1">
        <div className="flex h-full flex-col items-center justify-center gap-2">
          <div className="flex items-center gap-2">
            <IconFileSearch className="h-6 w-6" />
            <div className="text-xl font-semibold">{title}</div>
          </div>
          <div>{description}</div>
          <Link
            href="/"
            onClick={clearHomeScrollPosition}
            className={cn(buttonVariants(), 'mt-4 gap-2')}
          >
            <span>Home</span>
          </Link>
        </div>
      </div>
    </div>
  )
}
