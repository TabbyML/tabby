import { REPOSITORY_KIND_METAS } from '../constants'
import { NewProvider } from './components/new-page'

export function generateStaticParams() {
  return REPOSITORY_KIND_METAS.map(item => ({ kind: item.enum.toLowerCase() }))
}

export default function IndexPage() {
  return <NewProvider />
}
