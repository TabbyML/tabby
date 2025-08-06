import { graphql } from '@/lib/gql/generates'

export const updateBrandingSettingMutation = graphql(/* GraphQL */ `
  mutation updateBrandingSetting($input: BrandingSettingInput!) {
    updateBrandingSetting(input: $input)
  }
`)

export const brandingSettingQuery = graphql(/* GraphQL */ `
  query BrandingSetting {
    brandingSetting {
      brandingLogo
    }
  }
`)
