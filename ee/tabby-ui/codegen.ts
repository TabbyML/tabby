
import type { CodegenConfig } from '@graphql-codegen/cli';

const config: CodegenConfig = {
  overwrite: true,
  schema: "../tabby-schema/graphql/schema.graphql",
  documents: "./**/*.(tsx|ts)",
  ignoreNoDocuments: true,
  generates: {
    "lib/gql/generates/": {
      preset: "client",
      plugins: []
    },
    "lib/gql/generates/schema.json": {
      plugins: ['introspection'],
      config: {
        minify: true
      }
    }
  },
  hooks: { afterAllFileWrite: ['prettier --write'] }
};

export default config;
