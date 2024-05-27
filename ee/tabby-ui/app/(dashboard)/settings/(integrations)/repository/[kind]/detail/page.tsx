import { PROVIDER_KIND_METAS } from '../../constants'
import ProviderDetail from './components/detail-page'

export function generateStaticParams() {
  return PROVIDER_KIND_METAS.map(item => ({ kind: item.name }))
}

export default function IndexPage() {
  return <ProviderDetail />
}
