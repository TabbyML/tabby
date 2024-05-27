import { PROVIDER_KIND_METAS } from '../../constants'
import { NewProvider } from './components/new-page'

export function generateStaticParams() {
  return PROVIDER_KIND_METAS.map(item => ({ kind: item.name }))
}

export default function IndexPage() {
  return <NewProvider />
}
