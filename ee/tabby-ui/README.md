# Tabby UI

## 🤝 Contributing

### Local Setup
Full guide at [CONTRIBUTING.md](https://github.com/TabbyML/tabby/blob/main/CONTRIBUTING.md#local-setup)

### Running
During local development, we use Caddy to orchestrate Tabby-UI and local Tabby. We run both the Tabby-UI server and the local Tabby server simultaneously, using Caddy to forward frontend and backend requests to their respective servers, reducing the need for host and port configuration and taking advantage of the hot-reload feature of tabby-ui. 
The Caddy configuration file is located [here](https://github.com/TabbyML/tabby/blob/main/ee/tabby-webserver/development/Caddyfile).

Regarding the Tabby binary in production distribution, we do not start the tabby-ui server and Caddy server. Instead, tabby-ui is solely built and outputs static assets. Routing is configured within Tabby to distribute the static assets produced by tabby-ui.

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

#### 4. Start hacking
Now, you can open `http://localhost:8080` to see the tabby webserver!

---

You might also run `make dev` directly to execute the commands above simultaneously. (requires `tmux` and `tmuxinator`).
