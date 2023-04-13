const path = require('path');

module.exports = {
  target: 'node',
  mode: 'none',
  output: {
    filename: 'tabby.js',
    path: path.resolve(__dirname, 'dist'),
  },
  entry: './src/index.ts',
  resolve: {
    extensions: ['.ts', '.js'],
    preferRelative: true,
  },
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
};
