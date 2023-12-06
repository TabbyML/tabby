
import type { CodegenConfig } from '@graphql-codegen/cli';

const config: CodegenConfig = {
  overwrite: true,
  schema: "../tabby-webserver/graphql/schema.graphql",
  documents: "./**/*.(tsx|ts)",
  ignoreNoDocuments: true,
  generates: {
    "lib/gql/generates/": {
      preset: "client",
      plugins: []
    }
  },
  hooks: { afterAllFileWrite: ['prettier --write'] }
};

export default config;
