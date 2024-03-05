export * from './common'
export * from './chat'
export * from './repositories'

declare global {
  interface Window {
    _originFetch?: Window['fetch'] | undefined
  }
}
