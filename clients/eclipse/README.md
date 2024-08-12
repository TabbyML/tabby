# Tabby Plugin for Eclipse

**Note:** This project is still under development.

## Development

1. Install Eclipse with [PDE](https://projects.eclipse.org/projects/eclipse.pde). It is recommended to install the [Eclipse IDE for Eclipse Committers](https://www.eclipse.org/downloads/packages/release/2024-06/r/eclipse-ide-eclipse-committers).

2. Install [Node.js](https://nodejs.org/en/download/) >= 18. Install [pnpm](https://pnpm.io/installation), using [corepack](https://pnpm.io/installation#using-corepack) is recommended.

3. Clone Tabby repository, install dependencies.

```bash
git clone https://github.com/TabbyML/tabby.git
cd tabby
pnpm install
```

This will also build `tabby-agent` and copy it into `clients/eclipse/plugin/tabby-agent/`.

4. Import `clients/eclipse/plugin` and `clients/eclipse/feature` into Eclipse workspace.

5. Open `clients/eclipse/plugin/plugin.xml` in Eclipse, it should be open as a plugin project overview. In the `Testing` section, select `Launch an Eclipse application`.
