export * from './common'
export * from './chat'
export * from './repositories'
export * from './sso'

declare global {
  interface Window {
    _originFetch?: Window['fetch'] | undefined
  }
}
