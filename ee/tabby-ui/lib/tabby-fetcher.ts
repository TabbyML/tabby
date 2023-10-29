export default function fetcher(url: string): Promise<any> {
  if (process.env.NODE_ENV === 'production') {
    return fetch(url).then(x => x.json())
  } else {
    return fetch(`${process.env.NEXT_PUBLIC_TABBY_SERVER_URL}${url}`).then(x =>
      x.json()
    )
  }
}
