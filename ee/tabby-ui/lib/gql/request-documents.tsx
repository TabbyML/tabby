import {graphql} from './generates'

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
      gpuDevices
    }
  }
`)

export const getRegistrationTokenDocument = graphql(/* GraphQL */ `
  query GetRegistrationToken {
    registrationToken
  }
`)
