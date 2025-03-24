import { useCallback } from 'react'
import {
  ReadonlyURLSearchParams,
  usePathname,
  useRouter,
  useSearchParams
} from 'next/navigation'

type UpdateUrlComponentsOptions = {
  pathname?: string
  searchParams?: {
    set?: Record<string, string>
    del?: string | string[]
  }
  hash?: string
  replace?: boolean
}

export default function useRouterStuff() {
  const pathname = usePathname()
  const router = useRouter()
  const searchParams = useSearchParams()

  const getQueryString = useCallback(
    (kv?: Record<string, string>) => {
      const newParams = new URLSearchParams(searchParams.toString())
      if (kv) {
        Object.entries(kv).forEach(([k, v]) => newParams.set(k, v))
      }
      const queryString = newParams.toString()
      return queryString.length > 0 ? `?${queryString}` : ''
    },
    [searchParams]
  )

  const updateUrlComponents = useCallback(
    (options: UpdateUrlComponentsOptions) => {
      let newPath = resolveUrlComponents(
        options?.pathname || pathname,
        searchParams,
        options
      )

      if (options.replace) {
        router.replace(newPath)
      } else {
        router.push(newPath)
      }

      return newPath
    },
    [pathname, searchParams]
  )

  return {
    pathname,
    router,
    searchParams,
    getQueryString,
    updateUrlComponents
  }
}

function resolveUrlComponents(
  pathname: string,
  searchParams: ReadonlyURLSearchParams,
  options: UpdateUrlComponentsOptions
) {
  const set = options.searchParams?.set
  const del = options.searchParams?.del
  const newParams = new URLSearchParams(searchParams.toString())
  if (set) {
    Object.entries(set).forEach(([k, v]) => newParams.set(k, v))
  }
  if (del) {
    if (Array.isArray(del)) {
      del.forEach(k => newParams.delete(k))
    } else {
      newParams.delete(del)
    }
  }
  const queryString = newParams.toString()
  let newPath = pathname
  if (queryString.length > 0) {
    newPath += `?${queryString}`
  }
  if (options.hash) {
    newPath += `#${options.hash.replace(/^#/, '')}`
  }
  return newPath
}
