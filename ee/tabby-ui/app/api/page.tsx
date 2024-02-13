import { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'API'
}

export default function IndexPage() {
  return <iframe className="grow" src="/swagger-ui" />
}
