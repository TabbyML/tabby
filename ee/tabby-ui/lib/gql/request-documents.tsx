import { graphql } from './generates'

export const getAllWorkersDocument = graphql(/* GraphQL */ `
  query GetWorkers {
    workers {
      kind
      name
      addr
      device
      arch
      cpuInfo
      cpuCount
      cudaDevices
    }
  }
`)
