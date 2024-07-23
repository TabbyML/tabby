# Tabby UI

## 🤝 Contribuing
Thank you for your interest in contributing to Tabby! We appreciate all contributions. For a better experience and support, join us on [Slack](https://links.tabbyml.com/join-slack)!

### Local Setup
Full guide at [CONTRIBUTING.md](https://github.com/TabbyML/tabby/blob/main/CONTRIBUTING.md#local-setup)

### Get the Code
To begin contributing to Tabby, first clone the repository locally:

```
git clone --recurse-submodules https://github.com/TabbyML/tabby
```

If you have already cloned the repository, you could run the git submodule update --recursive --init command to fetch all submodules.

### Running

#### 1. Start the development frontend server

```
cd tabby/ee/tabby-ui
pnpm dev
```

#### 2. Start the development backend server

```
cargo run serve --port 8081
```

#### 3.Start the caddy server

```
make caddy
```

To enhance the local development experience, we use Caddy as a reverse proxy to forward requests during development, and the configuration file is located [here](https://github.com/TabbyML/tabby/blob/main/ee/tabby-webserver/development/Caddyfile)

#### 4. Start hacking
Now, you can open `http://localhost:8080` to see the tabby webserver!