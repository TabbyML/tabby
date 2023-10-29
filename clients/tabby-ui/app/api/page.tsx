import { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'API',
}

const serverUrl = process.env.NEXT_PUBLIC_TABBY_SERVER_URL || '';

export default function IndexPage() {
  return <iframe className='flex-grow' src={`${serverUrl}/swagger-ui`} />
}
