import { PROVIDER_KIND_METAS } from '../../constants'
import ProviderDetail from './components/provider-detail'

export function generateStaticParams() {
  return PROVIDER_KIND_METAS.map(item => ({ kind: item.name }))
}

export default function IndexPage() {
  return <ProviderDetail />
}
