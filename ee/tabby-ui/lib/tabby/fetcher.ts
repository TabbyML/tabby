export default function tokenFetcher([url, token]: [
  string,
  string
]): Promise<any> {
  const headers = new Headers()
  headers.append('authorization', `Bearer ${token}`)

  if (process.env.NODE_ENV !== 'production') {
    url = `${process.env.NEXT_PUBLIC_TABBY_SERVER_URL}${url}`
  }

  return fetch(url, { headers }).then(x => x.json())
}
