---
sidebar_position: 1
---

# Upgrade

:::caution
Before upgrade, make sure to [back up](../backup) the database.
:::


Tabby is a fast-evolving project, and we are constantly adding new features and fixing bugs. To keep up with the latest improvements, you should regularly upgrade your Tabby installation.

*Warning: Tabby does not support downgrade. Make sure to back up your meta data before upgrading.*

# Upgrade Procedure

The standard procedure for upgrading Tabby involves the following steps:

1. Back up the Tabby database.
2. Perform the upgrade
   1. If using docker, pull the latest image: `docker pull tabbyml/tabby`
   2. If using a standalone release, download it from the [releases page](https://github.com/TabbyML/tabby/releases) to replace the executable.
   3. Otherwise, just:
5. Restart Tabby.

That's it! You've successfully upgraded Tabby. If you encounter any issues, please consider joining our [slack community](https://links.tabbyml.com/join-slack) for help.
