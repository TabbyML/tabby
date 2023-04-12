const TypescriptDeclarationPlugin = require('typescript-declaration-webpack-plugin');

// Fix duplicated declaration like `export { ... } from "./..."` due to merge index.d.ts
class FixedTypescriptDeclarationPlugin extends TypescriptDeclarationPlugin {
  mergeDeclarations(declarationFiles) {
    const filtered = Object.fromEntries(Object.entries(declarationFiles).filter(([key]) => !key.endsWith('index.d.ts')));
    return super.mergeDeclarations(filtered);
  }
}

function getConfig(name) {
  let production = (name.indexOf('min') > -1);
  let config = {
    mode: production ? 'production' : 'development',
    entry: './generated/index.ts',
    resolve: {
      extensions: ['.ts', '.js']
    },
    output: {
      path: __dirname,
      filename: name + '.js',
      sourceMapFilename: name + '.map',
      library: 'Tabby',
      libraryTarget: 'umd',
      globalObject: 'this'
    },
    externals: {
      'axios': 'axios',
      'form-data': 'form-data',
    },
    devtool: 'source-map',
    module: {
      rules: [
        {
          test: /\.ts$/,
          exclude: /node_modules/,
          use: [
            {
              loader: 'ts-loader'
            }
          ]
        }
      ]
    },
    plugins: production ? [] : [
      new FixedTypescriptDeclarationPlugin({
        out: 'index.d.ts',
        removeComments: false,
      }),
    ]
  };
  return config;
}

module.exports = ['index', 'index.min'].map(getConfig);
