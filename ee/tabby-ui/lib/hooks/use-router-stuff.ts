import {
  ReadonlyURLSearchParams,
  usePathname,
  useRouter,
  useSearchParams
} from 'next/navigation'

type UpdateSearchParamsOptions = {
  set?: Record<string, string>
  del?: string | string[]
  replace?: boolean
  path?: string
}

export default function useRouterStuff() {
  const pathname = usePathname()
  const router = useRouter()
  const searchParams = useSearchParams()

  const getQueryString = (kv?: Record<string, string>) => {
    const newParams = new URLSearchParams(searchParams)
    if (kv) {
      Object.entries(kv).forEach(([k, v]) => newParams.set(k, v))
    }
    const queryString = newParams.toString()
    return queryString.length > 0 ? `?${queryString}` : ''
  }

  const updateSearchParams = (option: UpdateSearchParamsOptions) => {
    const newPath = resolveSearchParams(pathname, searchParams, option)

    if (option.replace) {
      router.replace(newPath)
    } else {
      router.push(newPath)
    }
  }

  const updatePathnameAndSearch = (
    pathname: string,
    option?: UpdateSearchParamsOptions
  ) => {
    let newPath = pathname
    if (option) {
      newPath = resolveSearchParams(pathname, searchParams, option)
    }

    if (option?.replace) {
      router.replace(newPath)
    } else {
      router.push(newPath)
    }
  }

  return {
    pathname,
    router,
    searchParams,
    updateSearchParams,
    getQueryString,
    updatePathnameAndSearch
  }
}

function resolveSearchParams(
  pathname: string,
  searchParams: ReadonlyURLSearchParams,
  option: UpdateSearchParamsOptions
) {
  const { set, del } = option
  const newParams = new URLSearchParams(searchParams)
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
  const newPath = `${pathname}${
    queryString.length > 0 ? `?${queryString}` : ''
  }`
  return newPath
}
